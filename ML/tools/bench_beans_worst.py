"""G3 worst case: bytes→dict latency of the bean candidate on the densest trainval images + synthetic dense piles.

Densest = top-N trainval images by n_beans from eval_beans per_image.csv (beans mode), original and JPEG≤1600.
Synthetic: plain background with 50/100/150/200 bean ellipses (stress the 150-blob cap). Notebook only.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roastml.api import load
from tools.bench_candidates import checked_image_path, fingerprint, web1600


def synthetic(n: int, size=(1600, 1200)) -> bytes:
    w, h = size
    img = np.full((h, w, 3), 235, np.uint8)
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    rx, ry = max(4, int(w / cols * 0.32)), max(3, int(h / rows * 0.25))
    for i in range(n):
        cx, cy = int((i % cols + .5) * w / cols), int((i // cols + .5) * h / rows)
        cv2.ellipse(img, (cx, cy), (rx, ry), 30, 0, 360, (110 - 30 * (i % 3), 72 - 18 * (i % 3), 45 - 10 * (i % 3)), -1)
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--per-image", type=Path, required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.data_dir.resolve()
    with args.per_image.open(newline="", encoding="utf-8") as f:
        per = [r for r in csv.DictReader(f) if r["n_beans"] not in ("", "None")]
    per.sort(key=lambda r: -int(r["n_beans"]))
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        man = {r["path"]: r for r in csv.DictReader(f)}
    dense = per[: args.top]
    pred = load(args.model)
    cases = {"densest_original": [], "densest_jpeg1600": []}
    for r in dense:
        raw = checked_image_path(root, man[r["path"]]).read_bytes()  # trainval only
        cases["densest_original"].append(raw)
        cases["densest_jpeg1600"].append(web1600(raw, "", 85)[0])
    for n in (50, 100, 150, 200):
        cases[f"synthetic_{n}"] = [synthetic(n)]
    out = {}
    for name, blobs in cases.items():
        times, n_beans = [], []
        for _ in range(args.repeat):
            for raw in blobs:
                t0 = time.perf_counter(); res = pred.predict_bytes(raw); times.append((time.perf_counter() - t0) * 1000)
                n_beans.append(res["n_beans"])
        out[name] = {"n_calls": len(times), "p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
                     "max": float(max(times)), "n_beans_seen": sorted({x for x in n_beans if x is not None})[-5:],
                     "null_n_beans": sum(x is None for x in n_beans)}
        print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    rep = {"model": fingerprint(args.model), "host": platform.processor(), "python": platform.python_version(),
           "densest_n_beans": [int(r["n_beans"]) for r in dense[:10]], "cases": out,
           "note": "notebook latency, not Pi evidence; trainval only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
