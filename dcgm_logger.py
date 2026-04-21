import subprocess
from datetime import datetime

_proc = None
_log_fh = None


def start_logging(filepath: str):
    global _proc, _log_fh
    _log_fh = open(filepath, "w")
    _proc = subprocess.Popen(
        ["sudo", "dcgmi", "dmon", "--host", "127.0.0.1:5555",
         "-e", "203,204,252,155,100", "-d", "200"],
        stdout=_log_fh,
        stderr=subprocess.DEVNULL,
    )


def write_marker(filepath: str, label: str):
    with open(filepath, "a") as f:
        f.write(f"# MARKER: {label} | {datetime.now().isoformat()}\n")
        f.flush()


def stop_logging():
    global _proc, _log_fh
    if _proc is not None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
        _proc = None
    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None
