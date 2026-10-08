"""Bench end-to-end bytes → dict (decode + smart crop + feature + predict) ผ่าน roastml.api เหมือนที่ FW เรียก

- ภาพ: manifest split=trainval สุ่มต่อ source ของ baseline (ค่าเริ่มต้น 50/source × 4 = 200) · agtron ใช้ไฟล์ภาพเต็มจากมือถือ
  (4000×3000+ = กรณี decode หนักสุด; ใช้วัดเวลาเท่านั้น ไม่ได้ประเมินความแม่น) · อ่าน bytes เข้า RAM ก่อนจับเวลา
- วัด: wall-clock ต่อภาพ + timing_ms ที่ api รายงาน (decode / ml / total) → p50 / p95 / max ต่อ source และรวม
- ขนาดไฟล์ (KB) และขนาดภาพ · RAM: RSS หลังโหลดโมเดล และ peak RSS ของ process
- เทียบต้นทุน decode JPEG ภาพใหญ่ (agtron): ไม่ใช้ draft · draft แบบเดิม (กรอบจัตุรัส) · draft ตามสัดส่วน (ปัจจุบัน)

ผล → ML/results/bench_<host>.json   ใช้: python -m tools.bench [--model models/current] [--per-source 50] [--host notebook]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import roastml
from roastml.api import load
from roastml.decode import MAX_SIDE, decode_image, draft_target
from roastml.paths import data_dir

ML_DIR = Path(__file__).resolve().parent.parent
RESULTS = ML_DIR / "results"
SEED = 20261006
from tools.train_baseline import FOLDS as SOURCES  # source เดียวกับ baseline (ไม่มี rf_hendi)


# ---------------------------------------------------------------- RAM
def rss_mb() -> dict[str, float | None]:
    """RSS ปัจจุบันและ peak ของ process นี้ (MB) · Windows ใช้ GetProcessMemoryInfo · Linux/Pi ใช้ /proc + getrusage"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return {"rss": c.WorkingSetSize / 2**20, "peak": c.PeakWorkingSetSize / 2**20}
        return {"rss": None, "peak": None}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        return {"rss": rss, "peak": peak}
    except (ImportError, OSError, ValueError):
        return {"rss": None, "peak": None}


# ---------------------------------------------------------------- decode variants (เทียบต้นทุน)
def _decode_variant(data: bytes, mode: str) -> tuple[int, int]:
    """mode: none | square (แบบเดิม) | aspect (ปัจจุบัน = draft_target) · คืนขนาดหลัง draft (ก่อน thumbnail)"""
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if mode == "square":
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
    elif mode == "aspect":
        img.draft("RGB", draft_target(w, h, MAX_SIDE))
    img.load()
    drafted = img.size
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return drafted


def stats_ms(v: list[float]) -> dict[str, float]:
    a = np.asarray(v, float)
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "max": round(float(a.max()), 1),
            "mean": round(float(a.mean()), 1)}


def pick_rows(root: Path, per_source: int) -> list[dict]:
    with open(root / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "trainval"]  # ไม่แตะ test
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        if r["label"] in ("light", "medium", "dark"):
            by[r["source"]].append(r)
    out = []
    for s in SOURCES:
        out += rng.sample(by[s], min(per_source, len(by[s])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model", default=str(ML_DIR / "models" / "current"))
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--host", default="notebook")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import cv2

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    rows = pick_rows(root, args.per_source)
    blobs = [(r["source"], (root / r["path"]).read_bytes()) for r in rows]
    sizes = []
    for _, b in blobs:
        with Image.open(io.BytesIO(b)) as im:
            sizes.append(im.size)

    mem0 = rss_mb()
    t0 = time.perf_counter()
    predictor = load(args.model)  # รวม warmup
    load_ms = (time.perf_counter() - t0) * 1e3
    mem_loaded = rss_mb()
    for _, b in blobs[:5]:  # อุ่นเครื่องเพิ่ม (cache ของ OpenCV/Pillow)
        predictor.predict_bytes(b)

    per = defaultdict(lambda: defaultdict(list))
    statuses = defaultdict(int)
    for (src, b) in blobs:
        t = time.perf_counter()
        r = predictor.predict_bytes(b)
        wall = (time.perf_counter() - t) * 1e3
        statuses[r["status"]] += 1
        for k, v in (("wall", wall), ("decode", r["timing_ms"]["decode"]), ("ml", r["timing_ms"]["ml"]),
                     ("total", r["timing_ms"]["total"])):
            per[src][k].append(v)
            per["ALL"][k].append(v)
    mem_end = rss_mb()

    # decode variants บน JPEG ใหญ่ของ agtron
    big = [b for (s, b), (w, h) in zip(blobs, sizes) if s == "agtron"]
    dec = {}
    for mode in ("none", "square", "aspect"):
        times, shapes = [], defaultdict(int)
        for b in big:
            t = time.perf_counter()
            shp = _decode_variant(b, mode)
            times.append((time.perf_counter() - t) * 1e3)
            shapes[f"{shp[0]}x{shp[1]}"] += 1
        dec[mode] = {"ms": stats_ms(times), "size_after_draft": dict(shapes)}

    kb = np.array([len(b) / 1024 for _, b in blobs])
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": args.host, "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "cv2_threads": cv2.getNumThreads(),
        "versions": {"roastml": roastml.__version__, "python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "pillow": Image.__version__},
        "model": predictor.info()["backend"], "model_load_ms_incl_warmup": round(load_ms, 1),
        "n_images": len(blobs), "per_source_n": {s: len(per[s]["wall"]) for s in SOURCES},
        "statuses": dict(statuses),
        "latency_ms": {s: {k: stats_ms(v) for k, v in d.items()} for s, d in per.items()},
        "file_kb": {"p50": round(float(np.percentile(kb, 50)), 1), "p95": round(float(np.percentile(kb, 95)), 1),
                    "max": round(float(kb.max()), 1)},
        "file_kb_by_source": {s: round(float(np.median([len(b) / 1024 for ss, b in blobs if ss == s])), 1)
                              for s in SOURCES},
        "image_px_by_source": {s: sorted({f"{w}x{h}" for (ss, _), (w, h) in zip(blobs, sizes) if ss == s})[:6]
                               for s in SOURCES},
        "ram_mb": {"before_load": mem0, "after_load": mem_loaded, "after_bench": mem_end},
        "decode_jpeg_agtron": dec,
        "note": "wall = เวลา predict_bytes ทั้งหมดใน process (ไม่รวมอัปโหลด/HTTP) · เครื่องนี้ไม่ใช่ Pi",
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"bench_{args.host}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a = report["latency_ms"]["ALL"]
    print(f"n={len(blobs)} wall p50 {a['wall']['p50']} p95 {a['wall']['p95']} max {a['wall']['max']} ms "
          f"(decode p50 {a['decode']['p50']} · ml p50 {a['ml']['p50']})")
    for s in SOURCES:
        L = report["latency_ms"][s]
        print(f"  {s:10s} wall p50 {L['wall']['p50']:7.1f} p95 {L['wall']['p95']:7.1f} · decode p50 {L['decode']['p50']:6.1f} "
              f"· ml p50 {L['ml']['p50']:6.1f} · {report['file_kb_by_source'][s]} KB")
    print("RAM MB:", {k: {kk: (round(vv, 1) if vv else vv) for kk, vv in v.items()} for k, v in report["ram_mb"].items()})
    print("decode agtron:", {m: (d["ms"]["p50"], d["ms"]["p95"], d["size_after_draft"]) for m, d in dec.items()})
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
