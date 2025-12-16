import ray
import pandas as pd
import time
import psutil
import logging
from io import StringIO
import os
import socket
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import threading
from google.cloud import storage

# ----------- Global Logging Setup (console + generic file) -----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("system_metrics_master.log"),
        logging.StreamHandler()
    ]
)

# ----------- Ray Cluster-Based Metrics Logger -----------

@ray.remote
class MetricsLogger:
    """
    Per-node monitor, running a background thread that logs every "interval" seconds.
    filename:    logs/worker_monitor_<hostname>_cycle<cycle_id>.log
    """

    def __init__(self, node_id, cycle_id, interval=1, max_seconds=None):
        self.node_id = node_id
        self.cycle_id = cycle_id
        self.interval = interval
        self.max_seconds = max_seconds
        self._stop_event = threading.Event()

        self.machine_name = socket.gethostname()
        os.makedirs("logs", exist_ok=True)

        # One logger per machine+cycle
        logger_name = f"worker_monitor_{self.machine_name}_cycle{self.cycle_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clean up any old handlers if this logger existed before
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        log_path = os.path.join(
            "logs",
            f"worker_monitor_{self.machine_name}_cycle{self.cycle_id}.log"
        )
        # new file for each cycle
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] "
                f"[Node {self.node_id} / {self.machine_name}] %(message)s"
            )
        )
        self.logger.addHandler(fh)

        # Start monitoring thread
        self.thread = threading.Thread(
            target=self._monitor_loop,
            name=f"MetricsLogger-{self.machine_name}-cycle{self.cycle_id}",
            daemon=True,
        )
        self.thread.start()

    def _monitor_loop(self):
        start_time = time.time()
        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                net = psutil.net_io_counters()
                disk = psutil.disk_io_counters()
                disk_usage = psutil.disk_usage('/')
                swap = psutil.swap_memory()
                virt = psutil.virtual_memory()

                # Temperature (if available)
                temperature_info = "N/A"
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps and temps['coretemp']:
                        cpu_temp = temps['coretemp'][0].current
                        temperature_info = f"CPU Temp: {cpu_temp}°C"
                except Exception:
                    pass

                self.logger.info(
                    "CPU: %.1f%%, Mem: %.1f%%, "
                    "NetSent: %.2fMB, NetRecv: %.2fMB, "
                    "DiskRead: %.2fMB, DiskWrite: %.2fMB, "
                    "DiskUsage: %.1f%%, "
                    "SwapUsed: %.2fMB, SwapTotal: %.2fMB, "
                    "VirtUsed: %.2fMB, VirtTotal: %.2fMB, %s",
                    cpu,
                    mem.percent,
                    net.bytes_sent / 1e6,
                    net.bytes_recv / 1e6,
                    disk.read_bytes / 1e6,
                    disk.write_bytes / 1e6,
                    disk_usage.percent,
                    swap.used / 1e6,
                    swap.total / 1e6,
                    virt.used / 1e6,
                    virt.total / 1e6,
                    temperature_info,
                )

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

            if self.max_seconds is not None and (time.time() - start_time) >= self.max_seconds:
                break

            time.sleep(self.interval)
        # cleanup
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        self.thread.join()


def start_cluster_logging(cycle_id, interval=1, max_seconds=None):
    """
    Start a MetricsLogger actor on each live node.
    Returns a list of (node_id, logger_actor_handle).
    """
    loggers = []
    for node in ray.nodes():
        if node["Alive"]:
            node_id = node["NodeID"]
            logger = MetricsLogger.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
            ).remote(node_id, cycle_id, interval, max_seconds)
            loggers.append((node_id, logger))
    return loggers


# ----------- Data and Training Logic -----------

def read_file_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.download_as_text()


def load_df_from_gcs(bucket_name, file_name, nrows=None):
    file_contents = read_file_from_gcs(bucket_name, file_name)
    df = pd.read_csv(StringIO(file_contents), index_col=0, nrows=nrows)
    return df


@ray.remote(num_cpus=1)
def train_single_model(model_name, model, X_train, X_test, y_train, y_test, node_id, cycle_id):
    os.makedirs("logs", exist_ok=True)
    model_log_path = f"logs/model_data_cycle{cycle_id}_node_{node_id}.csv"

    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    coeffs = getattr(model, 'coef_', 'N/A')
    intercept = getattr(model, 'intercept_', 'N/A')
    end_time = time.time()

    write_header = not os.path.exists(model_log_path)
    with open(model_log_path, "a") as f:
        if write_header:
            f.write("mse,model,start_time,end_time,cycle_id,node_id\n")
        f.write(f"{mse},{model.__class__.__name__},{start_time},{end_time},{cycle_id},{node_id}\n")

    return {
        "mse": mse,
        "model": model.__class__.__name__,
        "coefficients": coeffs,
        "intercept": intercept,
        "start_time": start_time,
        "end_time": end_time,
        "cycle_id": cycle_id,
        "node_id": node_id,
    }


def main(cycle_id: int, bucket_name: str, file_name: str, target_column: str):
    """
    Run one full experiment cycle:
    - start per-node logging
    - load data from GCS
    - train models in parallel
    - stop logging
    """
    print(f"\n===== Starting cycle {cycle_id} =====")

    # Start logging on each node with this cycle ID (every 1s)
    cluster_loggers = start_cluster_logging(cycle_id=cycle_id, interval=1)

    # Use node_id of first logger for tagging model logs (arbitrary choice)
    node_id, _ = cluster_loggers[0]

    # ---- Load data for THIS cycle (like Dask version) ----
    print(f"[cycle {cycle_id}] Loading DataFrame from GCS...")
    df = load_df_from_gcs(bucket_name, file_name)
    print(f"[cycle {cycle_id}] DataFrame loaded with shape {df.shape}")

    X = df.drop(columns=[target_column, "date", "City"])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"[cycle {cycle_id}] Train/test split done. "
        f"X_train: {X_train.shape}, X_test: {X_test.shape}, "
        f"y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    # Put big arrays into Ray object store for THIS cycle
    X_train_ref = ray.put(X_train)
    X_test_ref = ray.put(X_test)
    y_train_ref = ray.put(y_train)
    y_test_ref = ray.put(y_test)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=10, random_state=42, n_jobs=1
        ),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=10, learning_rate=0.1, random_state=42
        ),
    }

    print(f"[cycle {cycle_id}] Launching {len(models)} training tasks on the cluster...")

    # Launch one remote training task per model
    futures = {
        name: train_single_model.remote(
            name,
            model,
            X_train_ref,
            X_test_ref,
            y_train_ref,
            y_test_ref,
            node_id,
            cycle_id,
        )
        for name, model in models.items()
    }

    # Gather results
    results = {}
    for name, obj_ref in futures.items():
        res = ray.get(obj_ref)
        results[name] = res
        print(
            f"[cycle {cycle_id}] Model {name} finished "
            f"(node {res['node_id']}), MSE={res['mse']:.4f}"
        )

    print("\nModel Results:")
    for model_name, res in results.items():
        print(f"{model_name} (cycle {res['cycle_id']}, node {res['node_id']}):")
        print(f"  Coefficients: {res['coefficients']}")
        print(f"  Intercept: {res['intercept']}")
        print(f"  Mean Squared Error: {res['mse']}\n")

    # Clean up big Python objects on the driver for this cycle
    del df, X, y, X_train, X_test, y_train, y_test

    print(f"[cycle {cycle_id}] Stopping logger actors...")
    for _, logger in cluster_loggers:
        ray.get(logger.stop.remote())
    print(f"[cycle {cycle_id}] All logger actors stopped.")



if __name__ == "__main__":
    ray.shutdown()
    ray.init(address="auto", ignore_reinit_error=True)

    bucket_name = "experimentsdata"
    file_name = "merged_data.csv"
    target_column = "temperature_2m"

    NUM_CYCLES = 20

    for cycle in range(1, NUM_CYCLES + 1):
        main(cycle, bucket_name, file_name, target_column)

    print("Shutting down Ray after all cycles...")
    ray.shutdown()
