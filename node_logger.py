import os
import time
import socket
import argparse
import threading
import logging
import psutil
import signal
import sys

_stop = threading.Event()

def _handle_sigterm(_signum, _frame):
    _stop.set()

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

def run_logger(cycle_id: int, interval: int = 1, out_dir: str = "logs"):
    hostname = socket.gethostname()
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(f"worker_monitor_{hostname}_cycle{cycle_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # clear handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    log_path = os.path.join(out_dir, f"worker_monitor_{hostname}_cycle{cycle_id}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] [Node {hostname}] %(message)s"))

    logger.addHandler(fh)

    start_time = time.time()
    while not _stop.is_set():
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

            logger.info(
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
            logger.error("Error in monitor loop: %s", e)

        time.sleep(interval)

    # cleanup
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--out", type=str, default="logs")
    args = ap.parse_args()

    run_logger(args.cycle, args.interval, args.out)

if __name__ == "__main__":
    main()
