"""Diagnostic ก่อนรัน baseline (ราคาถูก) — ปัญหาอยู่ที่ label/ข้อมูล หรืออยู่ที่โมเดล

A. smart crop: % ที่ fallback เป็นภาพเต็มต่อ source + contact sheet overlay สุ่ม 24 ภาพ/source
   → $ROAST_DATA_DIR/contact_sheets/smartcrop/<source>.jpg (มีภาพจริง → อยู่นอก repo)
   overlay: กรอบเขียว = smart crop · สีแดงจาง = พิกเซลที่ใช้คิด feature · ข้อความ = mode/label
B. label scale ข้าม source: median L*/a*/b* ของบริเวณเมล็ด (smart crop / ROI) ต่อ source × label
   → ML/results/diag_lab_by_source_label.csv + diag_lab_boxplot.png + สรุปการทับกันใน diag_baseline.json (ตัวเลขล้วน)

ใช้ split=trainval + label ∈ {light, medium, dark} เท่านั้น (เหมือน baseline) · ใช้ feature cache เดียวกับ train_baseline
ใช้: python -m tools.diagnose_baseline [--workers 10]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from roastml.decode import decode_image
from roastml.features import FEATURES_ALL
from roastml.paths import CONTACT_SHEETS, data_dir
from roastml.segment import SegConfig, segment
from tools.train_baseline import CLASSES, FOLDS, extract_features, load_rows

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 20261006
N_SHEET = 24
# สี categorical (dataviz reference palette slot 1–3, ผ่าน validator โหมด light; aqua < 3:1 → มี legend + CSV)
COLORS = {"light": "#2a78d6", "medium": "#eb6834", "dark": "#1baf7a"}


def load_view(root: Path, r: dict) -> tuple[np.ndarray, bool]:
    """ภาพที่ baseline เห็นจริง (decode เส้นทางเดียวกับ Pi) · คืน (rgb, find_beans)"""
    img = decode_image((root / r["path"]).read_bytes())
    rgb = np.asarray(img.image)
    if r["roi"]:
        s = img.scale
        x1, y1, x2, y2 = (int(round(int(v) * s)) for v in r["roi"].split())
        return rgb[y1:y2, x1:x2], False
    return rgb, True


def overlay(rgb: np.ndarray, find_beans: bool, cfg: SegConfig, size: int = 220) -> tuple[Image.Image, str]:
    seg = segment(rgb, cfg, find_beans=find_beans)
    h, w = seg.lab.shape[:2]
    work = Image.fromarray(rgb).resize((w, h))
    a = np.asarray(work).astype(np.float32)
    m = seg.pixel_mask
    a[m] = a[m] * 0.55 + np.array([255, 0, 0]) * 0.45
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = seg.crop_box
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0), width=max(2, w // 150))
    im.thumbnail((size, size))
    return im, seg.mode


def contact_sheet(root: Path, rows: list[dict], cfg: SegConfig, out: Path, cols: int = 6, size: int = 220) -> None:
    tiles = []
    for r in rows:
        rgb, fb = load_view(root, r)
        im, mode = overlay(rgb, fb, cfg, size)
        tile = Image.new("RGB", (size, size + 16), "white")
        tile.paste(im, ((size - im.width) // 2, 16))
        ImageDraw.Draw(tile).text((2, 2), f"{r['label']} | {mode}", fill="black")
        tiles.append(tile)
    nrow = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * (size + 4), nrow * (size + 20)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (size + 4), (i // cols) * (size + 20)))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def iqr_overlaps(stats: dict) -> list[dict]:
    """คู่ (source, label) ต่าง source ต่าง label ที่ช่วง IQR ของ L_med ทับกัน · เรียงตามขนาดที่ทับ"""
    out = []
    for (s1, l1), (s2, l2) in combinations(sorted(stats), 2):
        if s1 == s2 or l1 == l2:
            continue
        a, b = stats[(s1, l1)]["L_med"], stats[(s2, l2)]["L_med"]
        ov = min(a["p75"], b["p75"]) - max(a["p25"], b["p25"])
        if ov > 0:
            out.append({"a": f"{s1}:{l1}", "b": f"{s2}:{l2}", "overlap_L": round(ov, 2),
                        "a_iqr": [a["p25"], a["p75"]], "b_iqr": [b["p25"], b["p75"]]})
    return sorted(out, key=lambda d: -d["overlap_L"])


def inversions(stats: dict) -> list[dict]:
    """ระดับที่ "เข้มกว่า" ของ source หนึ่ง แต่ median L* สูงกว่าระดับ "อ่อนกว่า" ของอีก source"""
    order = {"dark": 0, "medium": 1, "light": 2}
    out = []
    for (s1, l1), (s2, l2) in combinations(sorted(stats), 2):
        if s1 == s2 or l1 == l2:
            continue
        (sa, la), (sb, lb) = ((s1, l1), (s2, l2)) if order[l1] < order[l2] else ((s2, l2), (s1, l1))
        ma, mb = stats[(sa, la)]["L_med"]["median"], stats[(sb, lb)]["L_med"]["median"]
        if ma > mb:  # เข้มกว่าแต่สว่างกว่า
            out.append({"darker_label": f"{sa}:{la}", "L_median": ma, "lighter_label": f"{sb}:{lb}", "L_median_2": mb})
    return out


def boxplot(vals: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [("L_med", "median L* (0–100)"), ("a_med", "median a*"), ("b_med", "median b*")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, (fname, title) in zip(axes, feats):
        pos, data, cols = [], [], []
        for i, s in enumerate(FOLDS):
            for j, c in enumerate(CLASSES):
                v = vals.get((s, c), {}).get(fname)
                if v is None or not len(v):
                    continue
                pos.append(i * 4 + j)
                data.append(v)
                cols.append(COLORS[c])
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                        medianprops={"color": "#1a1a19", "linewidth": 1.5}, whiskerprops={"linewidth": 1},
                        capprops={"linewidth": 1})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_edgecolor("#fcfcfb")
            patch.set_linewidth(2)
        ax.set_ylabel(title, color="#3a3a38")
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xticks([i * 4 + 1 for i in range(len(FOLDS))], FOLDS)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c]) for c in CLASSES]
    axes[0].legend(handles, CLASSES, ncol=3, frameon=False, loc="upper right")
    # matplotlib ไม่มี font ไทย → ข้อความในกราฟเป็นอังกฤษ
    axes[0].set_title("Bean-region Lab per source x label (box = IQR, line = median, whiskers = 1.5 IQR)",
                      loc="left", fontsize=11)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    cfg = SegConfig()
    rows = load_rows(root)
    feats, _ = extract_features(root, rows, cfg, args.workers)
    key = lambda r: f"{r['md5']}|{r['roi']}"  # noqa: E731
    rows = [r for r in rows if not feats[key(r)]["err"]]

    # ---------------- A
    mode_tab = {}
    rng = random.Random(SEED)
    for s in FOLDS:
        rs = [r for r in rows if r["source"] == s]
        c = Counter(feats[key(r)]["mode"] for r in rs)
        mode_tab[s] = {"n": len(rs), "modes": dict(c), "fallback_full_pct": round(100 * c.get("full", 0) / len(rs), 1),
                       "by_label_full_pct": {lab: round(100 * sum(feats[key(r)]["mode"] == "full" for r in rs if r["label"] == lab)
                                                        / max(1, sum(r["label"] == lab for r in rs)), 1) for lab in CLASSES}}
        sample = rng.sample(rs, min(N_SHEET, len(rs)))
        sample.sort(key=lambda r: CLASSES.index(r["label"]))
        contact_sheet(root, sample, cfg, root / CONTACT_SHEETS / "smartcrop" / f"{s}.jpg")
        print(f"A {s:10s} {mode_tab[s]}")

    # ---------------- B
    idx = {n: FEATURES_ALL.index(n) for n in ("L_med", "a_med", "b_med")}
    vals: dict = defaultdict(dict)
    for s in FOLDS:
        for c in CLASSES:
            X = np.array([feats[key(r)]["x"] for r in rows if r["source"] == s and r["label"] == c])
            if len(X):
                for n, i in idx.items():
                    vals[(s, c)][n] = X[:, i]
    stats = {k: {n: {"median": round(float(np.median(v)), 2), "p25": round(float(np.percentile(v, 25)), 2),
                     "p75": round(float(np.percentile(v, 75)), 2), "n": int(len(v))} for n, v in d.items()}
             for k, d in vals.items()}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "diag_lab_by_source_label.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "label", "n"] + [f"{n}_{q}" for n in idx for q in ("p25", "median", "p75")])
        for s in FOLDS:
            for c in CLASSES:
                if (s, c) in stats:
                    st = stats[(s, c)]
                    w.writerow([s, c, st["L_med"]["n"]] + [st[n][q] for n in idx for q in ("p25", "median", "p75")])
    boxplot(vals, RESULTS / "diag_lab_boxplot.png")

    ov, inv = iqr_overlaps(stats), inversions(stats)
    report = {"seed": SEED, "A_smartcrop": mode_tab,
              "B_overlap_L_iqr_cross_source_cross_label": ov, "B_inversions_L_median": inv}
    (RESULTS / "diag_baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\nB median L* (p25–p75):")
    for s in FOLDS:
        print(f"  {s:10s} " + "  ".join(f"{c}={stats[(s, c)]['L_med']['median']:.1f} "
                                        f"({stats[(s, c)]['L_med']['p25']:.1f}–{stats[(s, c)]['L_med']['p75']:.1f})"
                                        for c in CLASSES if (s, c) in stats))
    print(f"\nIQR ทับกัน (ต่าง source ต่าง label): {len(ov)} คู่ · inversion ของ median: {len(inv)} คู่")
    for d in inv:
        print(f"  {d['darker_label']} (L {d['L_median']}) สว่างกว่า {d['lighter_label']} (L {d['L_median_2']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
