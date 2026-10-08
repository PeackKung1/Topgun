"""Bench end-to-end bytes → dict (decode + smart crop + feature + predict) ผ่าน roastml.api เหมือนที่ FW เรียก

- ภาพ: manifest split=trainval สุ่มต่อ source ของ baseline (ค่าเริ่มต้น 50/source × 4 = 200) · agtron ใช้ไฟล์ภาพเต็มจากมือถือ
  (4000×3000+ = กรณี decode หนักสุด; ใช้วัดเวลาเท่านั้น ไม่ได้ประเมินความแม่น) · อ่าน bytes เข้า RAM ก่อนจับเวลา
- 2 case ของ input:
    original : ไฟล์ต้นฉบับ
    web1600  : จำลองที่หน้าเว็บส่งจริง — หมุนตาม EXIF แล้วย่อด้านยาว ≤ 1600 px, JPEG quality 85 (Pillow)
               (canvas.toBlob(..., 0.85) ของเบราว์เซอร์ใช้ encoder คนละตัว ขนาดไฟล์/คุณภาพจะใกล้เคียง ไม่เท่ากันเป๊ะ)
- วัด 2 แบบต่อภาพ:
    (1) end-to-end: predictor.predict_bytes (wall + timing_ms ที่ api รายงาน)
    (2) แยก stage ด้วยฟังก์ชันชุดเดียวกับ LinearBackend: decode / segment / features / predict
- p50 / p95 / max ต่อ case × source และรวม
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


def stage_times(predictor, data: bytes) -> tuple[dict[str, float], str]:
    """เวลาแต่ละ stage (ms) ด้วยฟังก์ชันเดียวกับ LinearBackend.predict · คืน (เวลา, label ที่ได้)"""
    from roastml.features import pixel_stats
    from roastml.segment import segment

    be = predictor.backend
    t0 = time.perf_counter()
    img = decode_image(data)
    rgb = np.asarray(img.image)
    t1 = time.perf_counter()
    seg = segment(rgb, be.cfg)
    t2 = time.perf_counter()
    x = pixel_stats(seg.lab[seg.pixel_mask])
    t3 = time.perf_counter()
    p = be.model.proba(x)[0]
    t4 = time.perf_counter()
    ms = {"decode": (t1 - t0) * 1e3, "segment": (t2 - t1) * 1e3, "features": (t3 - t2) * 1e3,
          "predict": (t4 - t3) * 1e3, "sum": (t4 - t0) * 1e3}
    return ms, be.model.classes[int(np.argmax(p))]


def to_web1600(data: bytes, quality: int = 85) -> bytes:
    """จำลองหน้าเว็บ: หมุนตาม EXIF → ย่อด้านยาว ≤ 1600 → JPEG (ไม่มี EXIF)"""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1600, 1600))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model", default=str(ML_DIR / "models" / "current"))
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--host", default="notebook")
    ap.add_argument("--web-quality", type=int, default=85)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import cv2

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    rows = pick_rows(root, args.per_source)
    cases = {"original": [(r["source"], (root / r["path"]).read_bytes()) for r in rows]}
    cases["web1600"] = [(s_, to_web1600(b, args.web_quality)) for s_, b in cases["original"]]
    sizes = {c: [Image.open(io.BytesIO(b)).size for _, b in blobs] for c, blobs in cases.items()}

    mem0 = rss_mb()
    t0 = time.perf_counter()
    predictor = load(args.model)  # รวม warmup
    load_ms = (time.perf_counter() - t0) * 1e3
    mem_loaded = rss_mb()
    if not hasattr(predictor.backend, "cfg"):
        raise SystemExit("bench แยก stage รองรับเฉพาะ backend B0/B1 (LinearBackend)")
    for _, b in cases["original"][:5]:  # อุ่นเครื่องเพิ่ม (cache ของ OpenCV/Pillow)
        predictor.predict_bytes(b)

    lat: dict = {}
    statuses: dict = defaultdict(int)
    mismatch = 0
    for case, blobs in cases.items():
        per = defaultdict(lambda: defaultdict(list))
        for src_, b in blobs:
            t = time.perf_counter()
            r = predictor.predict_bytes(b)
            wall = (time.perf_counter() - t) * 1e3
            statuses[f"{case}:{r['status']}"] += 1
            st, lab = stage_times(predictor, b)
            mismatch += lab != r["label"]
            vals = {"wall": wall, "api_decode": r["timing_ms"]["decode"], "api_ml": r["timing_ms"]["ml"]} | st
            for k, v in vals.items():
                per[src_][k].append(v)
                per["ALL"][k].append(v)
        lat[case] = {s_: {k: stats_ms(v) for k, v in d.items()} for s_, d in per.items()}
    mem_end = rss_mb()

    # decode variants บน JPEG ใหญ่ของ agtron (original)
    big = [b for s_, b in cases["original"] if s_ == "agtron"]
    dec = {}
    for mode in ("none", "square", "aspect"):
        times, shapes = [], defaultdict(int)
        for b in big:
            t = time.perf_counter()
            shp = _decode_variant(b, mode)
            times.append((time.perf_counter() - t) * 1e3)
            shapes[f"{shp[0]}x{shp[1]}"] += 1
        dec[mode] = {"ms": stats_ms(times), "size_after_draft": dict(shapes)} if times else {}

    def kb_stats(blobs):
        kb = np.array([len(b) / 1024 for _, b in blobs])
        return {"p50": round(float(np.percentile(kb, 50)), 1), "p95": round(float(np.percentile(kb, 95)), 1),
                "max": round(float(kb.max()), 1),
                "by_source_p50": {s_: round(float(np.median([len(b) / 1024 for ss, b in blobs if ss == s_])), 1)
                                  for s_ in SOURCES}}

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": args.host, "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "cv2_threads": cv2.getNumThreads(),
        "versions": {"roastml": roastml.__version__, "python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "pillow": Image.__version__},
        "model": predictor.info()["backend"], "model_load_ms_incl_warmup": round(load_ms, 1),
        "n_images_per_case": len(rows), "per_source_n": {s_: sum(r["source"] == s_ for r in rows) for s_ in SOURCES},
        "web_case": {"long_side_max": 1600, "jpeg_quality": args.web_quality, "exif": "หมุนแล้ว ไม่มี EXIF"},
        "statuses": dict(statuses),
        "stage_label_mismatch_vs_api": mismatch,
        "latency_ms": lat,
        "file_kb": {c: kb_stats(b) for c, b in cases.items()},
        "image_px_by_source": {c: {s_: sorted({f"{w}x{h}" for (ss, _), (w, h) in zip(cases[c], sizes[c]) if ss == s_})[:6]
                                   for s_ in SOURCES} for c in cases},
        "ram_mb": {"before_load": mem0, "after_load": mem_loaded, "after_bench": mem_end,
                   "note": "RSS รวม bytes ของภาพทดสอบทั้งหมดที่โหลดไว้ใน RAM — ดูส่วนต่างเป็นหลัก"},
        "decode_jpeg_agtron_original": dec,
        "note": "wall = เวลา predict_bytes ทั้งหมดใน process (ไม่รวมอัปโหลด/HTTP) · stage วัดอีกรอบแยกกัน · เครื่องนี้ไม่ใช่ Pi",
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"bench_{args.host}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    keys = ("wall", "decode", "segment", "features", "predict")
    for case in cases:
        print(f"\n[{case}] file KB p50 {report['file_kb'][case]['p50']} · p50/p95 ms")
        print("  " + f"{'source':10s} " + " ".join(f"{k:>15s}" for k in keys))
        for s_ in list(SOURCES) + ["ALL"]:
            L = lat[case][s_]
            print("  " + f"{s_:10s} " + " ".join(f"{L[k]['p50']:7.1f}/{L[k]['p95']:7.1f}" for k in keys))
    print("\nlabel mismatch (stage vs api):", mismatch)
    print("RAM MB:", {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in report["ram_mb"].items() if isinstance(v, dict)})
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
