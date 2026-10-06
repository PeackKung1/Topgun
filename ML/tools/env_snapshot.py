"""บันทึก environment ที่ติดตั้งจริง (python + pip freeze + GPU) → ML/results/env_<name>.json

ใช้: python tools/env_snapshot.py [--name notebook]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def snapshot() -> dict:
    freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"]) or ""
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {
        "recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip": _run([sys.executable, "-m", "pip", "--version"]),
        "pip_freeze": [ln for ln in freeze.splitlines() if ln.strip()],
        "nvidia_smi": gpu,  # None = ไม่มี nvidia-smi
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="notebook", help="ชื่อเครื่อง เช่น notebook / pi")
    args = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"env_{args.name}.json"
    out.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
