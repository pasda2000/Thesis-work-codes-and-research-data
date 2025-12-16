import os
import pandas as pd
import numpy as np
import time
import logging
import threading
import psutil

from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from dask.diagnostics import ProgressBar
from dask import delayed, compute
from dask.distributed import Client
import dask.dataframe as dd



def start_worker_monitor(interval: int = 1, exp_id: int | None = None, max_seconds: int = 600):
    """
    Start (or restart) a monitoring thread on each worker.
    Behaviour:
    - If an old monitor is running, it is stopped and replaced.
    - Creates a new log file:
        ./worker_logs/worker_monitor_<hostname>_er<exp_id>.log
    - Auto-stops after max_seconds even if stop_worker_monitor is never called.
    """
    import socket
    import threading as _threading
    import time as _time

    old_thread = getattr(start_worker_monitor, "_monitor_thread", None)
    old_event = getattr(start_worker_monitor, "_stop_event", None)
    if old_thread is not None and old_thread.is_alive() and old_event is not None:
        old_event.set()
        old_thread.join()
        delattr(start_worker_monitor, "_monitor_thread")
        delattr(start_worker_monitor, "_stop_event")

    stop_event = threading.Event()
    start_worker_monitor._stop_event = stop_event  

    log_dir = "./worker_logs"
    os.makedirs(log_dir, exist_ok=True)

    worker_id = socket.gethostname()
    exp_suffix = "unknown" if exp_id is None else str(exp_id)
    log_path = os.path.join(log_dir, f"worker_monitor_{worker_id}_er{exp_suffix}.log")

    logger_name = f"worker_monitor_{worker_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Worker] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Starting worker monitor for exp={exp_suffix}, interval={interval}s, "
                f"max_seconds={max_seconds}")

    def monitor():
        end_time = _time.time() + max_seconds

        while not stop_event.is_set() and _time.time() < end_time:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            disk_read = disk.read_bytes
            disk_write = disk.write_bytes
            disk_usage = psutil.disk_usage("/")
            disk_percent = disk_usage.percent
            swap = psutil.swap_memory()
            swap_used = swap.used / 1e6
            swap_free = swap.free / 1e6
            swap_total = swap.total / 1e6
            temperature_info = "N/A"
            try:
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    cpu_temp = temps["coretemp"][0].current
                    temperature_info = f"CPU Temp: {cpu_temp}°C"
            except Exception as e:
                logging.error(f"Error reading temperature sensors on worker: {e}")

            virtual_memory = psutil.virtual_memory()
            virtual_total = virtual_memory.total / 1e6
            virtual_used = virtual_memory.used / 1e6
            virtual_free = virtual_memory.free / 1e6

            logger.info(
                f"[exp={exp_suffix}] CPU: {cpu}%, Mem: {mem}%, "
                f"Sent: {net.bytes_sent / 1e6:.2f}MB, Recv: {net.bytes_recv / 1e6:.2f}MB, "
                f"DiskRead: {disk_read / 1e6:.2f}MB, DiskWrite: {disk_write / 1e6:.2f}MB, "
                f"DiskUsage: {disk_percent}%, "
                f"SwapUsed: {swap_used:.2f}MB, SwapFree: {swap_free:.2f}MB, "
                f"SwapTotal: {swap_total:.2f}MB, "
                f"VirtMemTotal: {virtual_total:.2f}MB, VirtMemUsed: {virtual_used:.2f}MB, "
                f"VirtMemFree: {virtual_free:.2f}MB, "
                f"{temperature_info}"
            )

            stop_event.wait(interval)

        logger.info(f"Worker monitor exiting for exp={exp_suffix} "
                    f"(stop_event={stop_event.is_set()}, "
                    f"timed_out={_time.time() >= end_time})")

        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    monitor_thread = threading.Thread(
        target=monitor,
        daemon=True,
        name="worker_monitor_thread",
    )
    monitor_thread.start()
    start_worker_monitor._monitor_thread = monitor_thread  # type: ignore[attr-defined]

    return f"Monitoring thread started on worker (logging to {log_path})"


def stop_worker_monitor():
    """
    Stop the monitoring thread on each worker, if it exists.
    """
    monitor_thread = getattr(start_worker_monitor, "_monitor_thread", None)
    stop_event = getattr(start_worker_monitor, "_stop_event", None)

    if monitor_thread is None or stop_event is None or not monitor_thread.is_alive():
        return "No monitor thread was running."

    stop_event.set()
    monitor_thread.join(timeout=10)

    delattr(start_worker_monitor, "_monitor_thread")
    delattr(start_worker_monitor, "_stop_event")

    return "Stopped monitoring thread."



def main(id: int):
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(f"model_training{id}.log", mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False  

    logger.info("Logger configured correctly")

    def log_step(name, func, *args, **kwargs):
        start = time.perf_counter()
        logger.info(f"Starting: {name}")
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f"Finished: {name} in {duration:.2f} seconds")
        return result

    stop_event = threading.Event()

    def monitor_resources(interval: int = 1):
        while not stop_event.is_set():
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            disk_read = disk.read_bytes
            disk_write = disk.write_bytes
            disk_usage = psutil.disk_usage("/")
            disk_percent = disk_usage.percent
            swap = psutil.swap_memory()
            swap_used = swap.used / 1e6
            swap_free = swap.free / 1e6
            swap_total = swap.total / 1e6
            temperature_info = "N/A"
            try:
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    cpu_temp = temps["coretemp"][0].current
                    temperature_info = f"CPU Temp: {cpu_temp}°C"
            except Exception as e:
                logging.error(f"Error reading temperature sensors on master: {e}")

            # Virtual Memory
            virtual_memory = psutil.virtual_memory()
            virtual_total = virtual_memory.total / 1e6
            virtual_used = virtual_memory.used / 1e6
            virtual_free = virtual_memory.free / 1e6

            logger.info(
                f"Resource Usage — CPU: {cpu}%, Memory: {mem}%, "
                f"Sent: {net.bytes_sent / 1e6:.2f}MB, Received: {net.bytes_recv / 1e6:.2f}MB, "
                f"Disk Read: {disk_read / 1e6:.2f}MB, Disk Write: {disk_write / 1e6:.2f}MB, "
                f"Disk Usage: {disk_percent}%, "
                f"Swap Used: {swap_used:.2f}MB, Swap Free: {swap_free:.2f}MB, "
                f"Swap Total: {swap_total:.2f}MB, "
                f"Virtual Memory - Total: {virtual_total:.2f}MB, Used: {virtual_used:.2f}MB, Free: {virtual_free:.2f}MB, "
                f"{temperature_info}"
            )

            time.sleep(interval)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    client = Client(address="tcp://127.0.0.1:8786")
    print("Connected to Dask cluster")
    print("Dashboard link:", client.dashboard_link)
    logger.info("Connected to Dask cluster")

    client.run(start_worker_monitor, interval=1, exp_id=id)

    def get_log_file_path():
        import logging as _logging

        for handler in _logging.getLogger().handlers:
            if hasattr(handler, "baseFilename"):
                return handler.baseFilename
        return "No file handler configured"

    client.run(get_log_file_path)

    try:
        def download_csv_from_gcs(bucket_name, file_name, destination_path="downloaded_data.csv"):
            logger.info("⬇Downloading from GCS...")
            client_gcs = storage.Client()
            bucket = client_gcs.bucket(bucket_name)
            blob = bucket.blob(file_name)

            try:
                blob.download_to_filename(destination_path)
                logger.info(f"Downloaded to: {destination_path}")
            except Exception as e:  # noqa: BLE001
                logging.error(f"Error downloading file: {e}")
                raise

            if not os.path.exists(destination_path):
                raise FileNotFoundError(f"Downloaded file not found at {destination_path}")

            return destination_path

        bucket_name = "experimentsdata"
        file_name = "merged_data.csv"
        local_csv_path = log_step("Download CSV", download_csv_from_gcs, bucket_name, file_name)

        def read_csv_in_chunks(file_path, chunk_size=50 * 1024 * 1024):
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        def save_csv_on_worker(data_bytes, file_path="downloaded_data.csv"):
            with open(file_path, "wb") as f:
                f.write(data_bytes)
            return file_path

        logger.info("Reading file into memory and chunking...")
        csv_chunks = log_step("Read CSV chunks", list, read_csv_in_chunks(local_csv_path))
        csv_bytes = b"".join(csv_chunks)

        log_step("Distribute CSV to workers", client.run, save_csv_on_worker, data_bytes=csv_bytes)

        log_step("Load CSV into DataFrame", lambda: None)
        df_pd = pd.read_csv("downloaded_data.csv", index_col=0)
        df = dd.from_pandas(df_pd, npartitions=4)
        logger.info("DataFrame loaded")

        target_column = "temperature_2m"
        df = df.drop(columns=["date", "City"], errors="ignore")

        X = df.drop(columns=[target_column]).compute()
        y = df[target_column].compute()

        X_train, X_test, y_train, y_test = log_step(
            "Train-test split",
            train_test_split,
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        logger.info("Split completed")

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=5, random_state=42),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=5, learning_rate=0.1, random_state=42
            ),
        }
        logger.info("Models initialized")

        @delayed
        def train_model(name, model, X_train, y_train, X_test, y_test):
            logger.info(f"Training started for {name}")
            start_time = time.time()
            start = time.perf_counter()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            duration = time.perf_counter() - start
            logger.info(f"Finished training {name} in {duration:.2f} seconds | MSE: {mse:.4f}")
            end_time = time.time()
            return {
                "model": type(model).__name__,
                "mse": mse,
                "coefficients": getattr(model, "coef_", "N/A"),
                "intercept": getattr(model, "intercept_", "N/A"),
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
            }

        tasks = [
            train_model(name, model, X_train, y_train, X_test, y_test)
            for name, model in models.items()
        ]

        with ProgressBar():
            results = compute(*tasks)

        for result in results:
            logger.info(
                f"Result for {result['name']}: MSE={result['mse']}, Type={result['model']}, "
                f"Start_time={result['start_time']}, End_time={result['end_time']}"
            )
            print(f"\nModel: {result['name']}")
            print(f"  MSE: {result['mse']}")
            print(f"  Type: {result['model']}")
            print(f"  Coefficients: {result['coefficients']}")
            print(f"  Intercept: {result['intercept']}")

    finally:
        try:
            stop_event.set()
            monitor_thread.join()
        except Exception:
            pass

        try:
            client.run(stop_worker_monitor)
        except Exception:
            pass

        try:
            client.close()
        except Exception:
            pass

        logger.info("Monitoring stopped. Program complete.")
        logging.shutdown()


if __name__ == "__main__":
    for i in range(20):
        main(i)
