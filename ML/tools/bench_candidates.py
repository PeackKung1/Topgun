"""Benchmark any deployed backend, exclusively on trainval images.

One model per fresh process keeps peak RSS attributable to that candidate. Outputs
are immutable. Notebook results do not establish Raspberry Pi or phone latency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from roastml.api import load
from roastml.contract import LABELS
from roastml.paths import data_dir
from roastml.tabular_backend import model_file_path
from tools.bench import rss_mb

SOURCES = ("ontoum224", "rf_robusta", "rf_boos", "agtron")
SEED = 20261006


def fingerprint(model_dir: Path) -> dict:
    card_path = Path(model_dir) / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    names = ["model_card.json", card.get("model_file", "model.onnx" if card.get("backend") == "onnx_rgb" else "model.json")]
    names += list(card.get("external_data_files", []))
    files, digest = {}, hashlib.sha256()
    for name in sorted(set(names)):
        p = model_file_path(model_dir, name)
        blob = p.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        files[name] = {"sha256": sha, "bytes": len(blob)}
        digest.update(name.encode("utf-8") + b"\0" + bytes.fromhex(sha))
    return {"artifact_sha256": digest.hexdigest(), "files": files,
            "size_bytes": sum(f["bytes"] for f in files.values()), "card": card}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pillow", "opencv-python-headless", "onnxruntime", "roastml"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def runtime_provenance() -> dict:
    import cv2
    import roastml
    root = Path(roastml.__file__).parent
    names = ("api.py", "decode.py", "rgb_views.py", "features.py", "segment.py", "linear_model.py",
             "linear_backend.py", "tabular_backend.py", "onnx_backend.py")
    return {"source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            "opencv_threads": cv2.getNumThreads(),
            "workload_note": "No system-wide CPU-idle constraint is enforced; control background load for latency comparison. Notebook timings are not Pi evidence."}


def hardware() -> dict:
    out = {"architecture": platform.machine(), "os": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "python": platform.python_version(), "board_model": None, "ram_bytes": None}
    if sys.platform.startswith("linux"):
        model = Path("/proc/device-tree/model")
        if model.is_file():
            out["board_model"] = model.read_bytes().rstrip(b"\0").decode("utf-8", "replace")
        try:
            out["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
        try:
            out["os_release"] = platform.freedesktop_os_release()
        except OSError:
            pass
    elif sys.platform == "win32":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_pagefile",
                                                       "available_pagefile", "total_virtual", "available_virtual", "available_extended")]
        value = Memory(); value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            out["ram_bytes"] = value.total_physical
    return out


def pick_rows(root: Path, per_source: int = 50, seed: int = SEED) -> list[dict]:
    by = defaultdict(list)
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "trainval":
                continue
            if row.get("source") in SOURCES and row.get("label") in LABELS:
                by[row["source"]].append(row)
    rng = random.Random(seed)
    selected = []
    for source in SOURCES:
        selected += rng.sample(by[source], min(per_source, len(by[source])))
    return selected


def checked_image_path(root: Path, row: dict) -> Path:
    if row.get("split") != "trainval":
        raise ValueError("benchmark refuses any image outside split=trainval")
    p = (root / row["path"]).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("manifest image path escapes data root")
    if row["source"] == "agtron" and not row.get("roi"):
        raise ValueError("Agtron requires ROI metadata before reading its image")
    return p


def web1600(raw: bytes, roi: str = "", quality: int = 84) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(raw)) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        width, height = rgb.size
        rgb.thumbnail((1600, 1600))
        mapped = ""
        if roi:
            coords = [float(v) for v in roi.split()]
            if len(coords) != 4:
                raise ValueError("ROI requires four coordinates")
            sx, sy = rgb.width / width, rgb.height / height
            mapped = " ".join(str(round(v * s)) for v, s in zip(coords, [sx, sy, sx, sy]))
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=quality)
        return buf.getvalue(), mapped


def stats(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {"n": len(values), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean())}


def benchmark(model: Path, root: Path, rows: list[dict], *, quality: int = 84, host: str = "notebook") -> dict:
    if not rows:
        raise ValueError("no trainval rows selected")
    # Validate every row before reading any image bytes.
    paths = [checked_image_path(root, row) for row in rows]
    original = [(row, path.read_bytes()) for row, path in zip(rows, paths)]
    web = [(row | {"roi": roi}, raw) for row, blob in original
           for raw, roi in [web1600(blob, row.get("roi", ""), quality)]]
    mem_before = rss_mb()
    identity = fingerprint(model)
    t0 = time.perf_counter()
    predictor = load(model)
    load_ms = (time.perf_counter() - t0) * 1000
    mem_loaded = rss_mb()
    for row, raw in original[:5]:
        predictor.predict_bytes(raw, source=row["source"], roi=row.get("roi", ""))
    report_cases = {}
    per_image = []
    for name, blobs in (("original", original), ("jpeg1600", web)):
        stage_samples = defaultdict(lambda: defaultdict(list))
        statuses, view_counts = Counter(), Counter()
        for row, raw in blobs:
            t0 = time.perf_counter()
            result, stages = predictor.predict_profile(raw, source=row["source"], roi=row.get("roi", ""))
            wall = (time.perf_counter() - t0) * 1000
            stages["wall"] = wall
            statuses[result["status"]] += 1
            view_counts[str(stages.get("n_views"))] += 1
            record = {"case": name, "path": row["path"], "source": row["source"], "status": result["status"],
                      "label": result["label"], "input_bytes": len(raw), "stages_ms": stages}
            per_image.append(record)
            for key, value in stages.items():
                if key == "n_views":
                    continue
                stage_samples[row["source"]][key].append(value)
                stage_samples["ALL"][key].append(value)
        report_cases[name] = {"latency_ms": {src: {stage: stats(values) for stage, values in table.items()}
                                              for src, table in stage_samples.items()},
                              "statuses": dict(statuses), "n_views": dict(view_counts)}
    return {"created_unix": time.time(), "host_label": host, "hardware": hardware(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "versions": versions(),
            "runtime_provenance": runtime_provenance(),
            "model": identity, "predictor_info": predictor.info(), "load_ms_including_warmup": load_ms,
            "n_images_per_case": len(rows), "manifest_sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest(),
            "sample_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "counts_by_source": dict(Counter(r["source"] for r in rows)), "cases": report_cases,
            "ram_mb": {"before_load": mem_before, "after_load": mem_loaded, "after_benchmark": rss_mb()},
            "memory_scope": "process RSS and process high-water mark; includes both input cases already in RAM",
            "jpeg_case": {"max_side": 1600, "quality": quality, "encoder": "Pillow; differs from browser canvas"},
            "metadata_policy": "Agtron mandatory post-EXIF ROI, no detection boxes for holdout/inference",
            "per_image": per_image, "verification": {"pi_latency": "unverified", "fw_e2e": "unverified"},
            "note": "bytes→dict only, excludes HTTP/upload; host_label is descriptive and does not prove hardware identity"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--jpeg-quality", type=int, default=84)
    ap.add_argument("--host", default="notebook")
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error("output already exists; choose a new immutable run file")
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    rows = pick_rows(root, args.per_source, args.seed)
    if len(rows) < 200:
        ap.error("benchmark requires at least 200 trainval images per input case")
    report = benchmark(args.model, root, rows, quality=args.jpeg_quality, host=args.host)
    report["seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "artifact_sha256": report["model"]["artifact_sha256"],
                      "p95_ms": {case: value["latency_ms"]["ALL"]["wall"]["p95"] for case, value in report["cases"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
