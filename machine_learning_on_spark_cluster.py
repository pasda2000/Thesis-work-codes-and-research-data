"""sparklogger.py

Spark version of the Ray/Dask regression experiment with per-cycle logging.

What this does:
- Runs multiple cycles.
- Each cycle:
  - Starts a background metrics logger on EVERY cluster node (driver + worker nodes)
    writing: logs/worker_monitor_<hostname>_cycle<cycle>.log
  - Downloads the CSV from GCS (same as Ray/Dask approach), loads into Spark DF
  - Trains the same family of models (Spark ML equivalents):
      LinearRegression, Ridge, Lasso, RandomForest, DecisionTree, GBTRegressor
  - Writes model metrics per cycle to:
      logs/model_data_cycle<cycle>_node_<hostname>.csv
  - Stops the per-node loggers

IMPORTANT:
- Spark executors are JVM processes; long-lived Python threads do not persist on executors.
  So per-node logging is implemented by starting a small Python process on each node via SSH.

Requirements:
- Passwordless SSH from the driver host to every worker host (same username).
- node_logger.py must exist at ~/node_logger.py on every node.
- google-cloud-storage must be installed (pip install google-cloud-storage).
- psutil must be installed on every node (pip install psutil).

Run (recommended):
  $SPARK_HOME/bin/spark-submit --master spark://<MASTER_IP>:7077 sparklogger.py

"""

import os
import time
import socket
import threading
import logging
import subprocess
from typing import Dict, List, Set, Optional

import psutil

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    DecisionTreeRegressor,
    GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator

from google.cloud import storage


# ---------------------------------------------------------
#  Driver metrics logger (thread)
# ---------------------------------------------------------

class MetricsLogger:
    """Background thread logging system metrics every `interval` seconds."""

    def __init__(self, node_id: str, cycle_id: int, interval: int = 1, max_seconds: Optional[int] = None):
        self.node_id = node_id
        self.cycle_id = cycle_id
        self.interval = interval
        self.max_seconds = max_seconds
        self._stop_event = threading.Event()

        self.machine_name = socket.gethostname()
        os.makedirs("logs", exist_ok=True)

        logger_name = f"worker_monitor_{self.machine_name}_cycle{self.cycle_id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Remove old handlers if they exist
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        log_path = os.path.join("logs", f"worker_monitor_{self.machine_name}_cycle{self.cycle_id}.log")
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] "
                f"[Node {self.node_id} / {self.machine_name}] %(message)s"
            )
        )
        self.logger.addHandler(fh)

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
                disk_usage = psutil.disk_usage("/")
                swap = psutil.swap_memory()
                virt = psutil.virtual_memory()

                temperature_info = "N/A"
                try:
                    temps = psutil.sensors_temperatures()
                    if "coretemp" in temps and temps["coretemp"]:
                        cpu_temp = temps["coretemp"][0].current
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


def start_driver_logging(cycle_id: int, interval: int = 1) -> MetricsLogger:
    hostname = socket.gethostname()
    return MetricsLogger(node_id=f"driver-{hostname}", cycle_id=cycle_id, interval=interval)


# ---------------------------------------------------------
#  Remote node loggers (SSH-launched processes)
# ---------------------------------------------------------

def _run_ssh(host: str, command: str) -> bool:
    try:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", host, command],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] SSH failed for {host}: {e}")
        return False


def get_executor_hosts(spark: SparkSession) -> List[str]:
    sc = spark.sparkContext
    jmap = sc._jsc.sc().getExecutorMemoryStatus()

    driver_hostnames = set()
    driver_hostnames.add(socket.gethostname())
    driver_hostnames.add(socket.getfqdn())

    # If Spark knows driver host, exclude that too
    try:
        dh = sc.getConf().get("spark.driver.host")
        if dh:
            driver_hostnames.add(dh)
    except Exception:
        pass

    hosts: Set[str] = set()
    it = jmap.keySet().iterator()
    while it.hasNext():
        key_str = str(it.next())
        if key_str.startswith("driver"):
            continue

        host = key_str.split(":")[0]
        if not host or host == "driver":
            continue

        # Skip driver/master host to avoid ssh-to-self
        if host in driver_hostnames:
            continue

        hosts.add(host)

    return sorted(hosts)



def start_node_loggers_via_ssh(
    hosts: List[str],
    cycle_id: int,
    remote_python: str = "~/spark-env/bin/python",
    remote_logger_path: str = "~/node_logger.py",
    out_dir: str = "logs",
    interval: int = 1,
) -> Dict[str, str]:
    """Start node_logger.py on each host in background; return host->pidfile."""
    pidfiles: Dict[str, str] = {}

    for host in hosts:
        pidfile = f"/tmp/node_logger_cycle_{cycle_id}.pid"
        pidfiles[host] = pidfile

        cmd = (
            f"mkdir -p {out_dir} ; "
            f"nohup {remote_python} {remote_logger_path} "
            f"--cycle {cycle_id} --interval {interval} --out {out_dir} "
            f"> /tmp/node_logger_cycle_{cycle_id}.out 2>&1 & "
            f"echo $! > {pidfile}"
        )
        if _run_ssh(host, cmd):
             pidfiles[host] = pidfile

    return pidfiles


def stop_node_loggers_via_ssh(pidfiles: Dict[str, str]) -> None:
    """Stop node_logger.py on each host via pidfile."""
    for host, pidfile in pidfiles.items():
        cmd = (
            f"if [ -f {pidfile} ]; then "
            f"  PID=$(cat {pidfile}) ; "
            f"  kill -TERM $PID 2>/dev/null || true ; "
            f"  rm -f {pidfile} ; "
            f"fi"
        )
        _run_ssh(host, cmd)


# ---------------------------------------------------------
#  GCS download (Ray/Dask-style: re-download each cycle)
# ---------------------------------------------------------

def download_csv_from_gcs(bucket_name: str, file_name: str, local_dir: str = "/tmp/spark-data") -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, file_name)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(local_path)

    return local_path

def ensure_csv_on_hosts_via_ssh(hosts: List[str], bucket: str, name: str, local_path: str):
    # Uses gsutil on the remote hosts. If you don't have gsutil installed, tell me and we’ll use python GCS.
    remote_dir = os.path.dirname(local_path)
    for host in hosts:
        cmd = (
            f"mkdir -p {remote_dir} && "
            f"gsutil -q cp gs://{bucket}/{name} {local_path}"
        )
        ok = _run_ssh(host, cmd)  # use your safe _run_ssh that returns True/False
        if not ok:
            print(f"[WARN] could not copy CSV to {host}")

# ---------------------------------------------------------
#  One cycle of Spark ML training
# ---------------------------------------------------------

def run_spark_cycle(
    spark: SparkSession,
    cycle_id: int,
    bucket_name: str,
    file_name: str,
    target_column: str,
    interval_seconds: int = 1,
) -> None:

    print(f"\n===== Starting Spark cycle {cycle_id} =====")

    # Force executor allocation early so we can discover hosts.
    # (Without an action, Spark may not spin up executors yet.)
    spark.range(1).count()

    # Start logging on all nodes
    driver_logger = start_driver_logging(cycle_id=cycle_id, interval=interval_seconds)

    hosts = get_executor_hosts(spark)
    print(f"[cycle {cycle_id}] Executor hosts discovered: {hosts}")

    pidfiles: Dict[str, str] = {}
    if hosts:
        pidfiles = start_node_loggers_via_ssh(
            hosts=hosts,
            cycle_id=cycle_id,
            remote_python="~/spark-env/bin/python",
            remote_logger_path="~/node_logger.py",
            out_dir="logs",
            interval=interval_seconds,
        )

    try:
        # Load data fresh each cycle (download from GCS)
        print(f"[cycle {cycle_id}] Downloading CSV from GCS...")
        local_csv_path = download_csv_from_gcs(bucket_name, file_name)
        print(f"[cycle {cycle_id}] Local CSV path: {local_csv_path}")

        ensure_csv_on_hosts_via_ssh(hosts, bucket_name, file_name, local_csv_path)

        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(f"file:{local_csv_path}")
        )

        rows = df.count()
        cols = len(df.columns)
        print(f"[cycle {cycle_id}] DataFrame loaded with {rows} rows and {cols} columns")

        # Feature columns
        exclude_cols = {target_column, "date", "City"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        print(f"[cycle {cycle_id}] Using {len(feature_cols)} feature columns")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        assembled_df = assembler.transform(df).select("features", target_column)

        train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)
        print(f"[cycle {cycle_id}] Train count: {train_df.count()}, Test count: {test_df.count()}")

        evaluator = RegressionEvaluator(
            labelCol=target_column,
            predictionCol="prediction",
            metricName="mse",
        )

        # Same algorithms as Ray/Dask (Spark ML equivalents)
        models = [
            (
                "LinearRegression",
                LinearRegression(featuresCol="features", labelCol=target_column, maxIter=100),
            ),
            (
                "Ridge",
                LinearRegression(
                    featuresCol="features",
                    labelCol=target_column,
                    maxIter=100,
                    regParam=1.0,
                    elasticNetParam=0.0,
                ),
            ),
            (
                "Lasso",
                LinearRegression(
                    featuresCol="features",
                    labelCol=target_column,
                    maxIter=100,
                    regParam=0.1,
                    elasticNetParam=1.0,
                ),
            ),
            (
                "RandomForest",
                RandomForestRegressor(
                    featuresCol="features",
                    labelCol=target_column,
                    numTrees=10,
                    maxDepth=10,
                ),
            ),
            (
                "DecisionTree",
                DecisionTreeRegressor(featuresCol="features", labelCol=target_column, maxDepth=10),
            ),
            (
                "GradientBoosting",
                GBTRegressor(featuresCol="features", labelCol=target_column, maxIter=10),
            ),
        ]

        hostname = socket.gethostname()
        os.makedirs("logs", exist_ok=True)
        model_log_path = f"logs/model_data_cycle{cycle_id}_node_{hostname}.csv"

        if not os.path.exists(model_log_path):
            with open(model_log_path, "w") as f:
                f.write("mse,model,start_time,end_time,cycle_id,node_id\n")

        print(f"[cycle {cycle_id}] Training {len(models)} models...")

        for model_name, estimator in models:
            print(f"[cycle {cycle_id}] Fitting {model_name} ...")
            start_time = time.time()

            fitted = estimator.fit(train_df)
            predictions = fitted.transform(test_df)
            mse = evaluator.evaluate(predictions)

            end_time = time.time()

            # Log to CSV
            with open(model_log_path, "a") as f:
                f.write(f"{mse},{model_name},{start_time},{end_time},{cycle_id},{hostname}\n")

            # Print similar to Ray/Dask
            print(f"[cycle {cycle_id}] {model_name}: MSE={mse:.6f} (node {hostname})")

    finally:
        # Stop remote node loggers
        if pidfiles:
            print(f"[cycle {cycle_id}] Stopping remote node loggers...")
            stop_node_loggers_via_ssh(pidfiles)

        # Stop driver logger thread
        print(f"[cycle {cycle_id}] Stopping driver logger thread...")
        driver_logger.stop()
        print(f"[cycle {cycle_id}] Logging stopped.")


# ---------------------------------------------------------
#  Multi-cycle runner
# ---------------------------------------------------------

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("SparkClusterWithLogs")
        .getOrCreate()
    )

    bucket_name = "experimentsdata"
    file_name = "merged_data.csv"
    target_column = "temperature_2m"

    NUM_CYCLES = 20

    for cycle in range(16, NUM_CYCLES + 1):
        run_spark_cycle(
            spark=spark,
            cycle_id=cycle,
            bucket_name=bucket_name,
            file_name=file_name,
            target_column=target_column,
            interval_seconds=1,
        )

    print("Shutting down Spark after all cycles...")
    spark.stop()
