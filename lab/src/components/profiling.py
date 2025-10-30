"""System profiling for resource monitoring during experiments."""
import os
import psutil
import threading
import time
import shutil
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path


GPU_UTIL_RE = re.compile(r"GPU Active residency:\s+(\d+)%")
GPU_PWR_RE = re.compile(r"GPU Power:\s+(\d+)\s+mW")


class Profiler:
    """System resource profiler with optional GPU metrics."""

    def __init__(self, interval_s=5.0, use_powermetrics=False):
        """Initialize profiler.

        Args:
            interval_s: Sampling interval in seconds
            use_powermetrics: Try to use powermetrics for GPU stats (requires sudo)
        """
        self.interval = interval_s
        self.rows = []
        self._stop = False
        self._thr = None
        self.use_pm = use_powermetrics and shutil.which("powermetrics")

        if use_powermetrics and not shutil.which("powermetrics"):
            print("Warning: 'powermetrics' enabled but binary not found in PATH.")
        if use_powermetrics and os.geteuid() != 0:
            print("Warning: 'powermetrics' requires sudo. GPU stats may fail.")

    def start(self):
        """Start profiling in background thread."""
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        """Stop profiling."""
        self._stop = True
        if self._thr:
            self._thr.join()

    def dump(self, path: Path):
        """Save profiling data to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.rows, f, indent=2)

    def _sample_pm(self):
        """Sample GPU stats from powermetrics."""
        try:
            # Note: Requires sudo access configured for powermetrics
            out = subprocess.check_output(
                ["sudo", "powermetrics", "-n", "1", "--samplers", "gpu_power"],
                timeout=3,
            ).decode()
            u = GPU_UTIL_RE.search(out)
            p = GPU_PWR_RE.search(out)
            return {
                "gpu_util_pct": int(u.group(1)) if u else None,
                "gpu_power_mw": int(p.group(1)) if p else None,
            }
        except Exception:
            # Fail silently if sudo fails or binary missing
            return {"gpu_util_pct": None, "gpu_power_mw": None}

    def _loop(self):
        """Profiling loop."""
        while not self._stop:
            row = {
                "t": datetime.utcnow().isoformat(),
                "rss_bytes": psutil.Process().memory_info().rss,
                "cpu_pct": psutil.cpu_percent(interval=None),
            }
            if self.use_pm:
                row.update(self._sample_pm())

            self.rows.append(row)
            time.sleep(self.interval)
