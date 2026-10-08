"""Error analysis ของ LOSO fold หนึ่ง: ภาพที่ label_orig = X แต่ถูกทาย = Y (ดูอย่างเดียว ห้ามใช้จูน)

- เทรนโมเดลแบบเดียวกับ train_baseline (train view ของ source อื่นทั้งหมด) แล้วทำนาย source ที่ hold out ด้วย pipeline view
- contact sheet: ภาพที่ผิด + ภาพกลุ่มเดียวกันที่ทายถูก (เทียบ) พร้อม overlay พิกเซลที่ใช้คิด feature
  → $ROAST_DATA_DIR/contact_sheets/errors/<ชื่อ>.jpg (มีภาพจริง → ห้าม commit)
- สรุปตัวเลข (ไม่มีชื่อไฟล์): device / flash / L*, a*, b* / prob → ML/results/error_<ชื่อ>.json
- รายชื่อไฟล์ → $ROAST_DATA_DIR/cache/error_<ชื่อ>.csv

ใช้: python -m tools.error_analysis --source agtron --label-orig agtron_25 --pred light [--model B1 --fs Lab_hist --C 0.01]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from roastml.features import FEATURES_ALL, FEATURE_SETS, select
from roastml.linear_model import LinearSoftmax
from roastml.paths import CACHE, CONTACT_SHEETS, RAW, data_dir
from roastml.segment import SegConfig
from tools.diagnose_baseline import load_view, overlay
from tools.train_baseline import CLASSES, attach_boxes, extract_features, fit_b1, load_rows, row_key

RESULTS = Path(__file__).resolve().parent.parent / "results"


def agtron_flash(root: Path) -> dict[str, str]:
    """ชื่อไฟล์กาแฟ → flash (0/1) จาก photos.csv"""
    p = root / RAW / "agtron" / "photos.csv"
    if not p.is_file():
        return {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {r["name_coffee"]: r["flash"] for r in csv.DictReader(f, delimiter=";")}


def tile(root: Path, r: dict, cfg: SegConfig, probs: np.ndarray, x: np.ndarray, size: int = 240) -> Image.Image:
    rgb, fb = load_view(root, r)
    im, mode = overlay(rgb, fb, cfg, size)
    t = Image.new("RGB", (size, size + 44), "white")
    t.paste(im, ((size - im.width) // 2, 44))
    d = ImageDraw.Draw(t)
    f = dict(zip(FEATURES_ALL, x))
    d.text((2, 2), f"{r['device'].split(':')[-1]} | flash={r.get('flash', '?')} | {mode}", fill="black")
    d.text((2, 15), f"L*={f['L_med']:.1f} a*={f['a_med']:.1f} b*={f['b_med']:.1f}", fill="black")
    d.text((2, 28), " ".join(f"{c[0]}={p:.2f}" for c, p in zip(CLASSES, probs)), fill="black")
    return t


def sheet(tiles: list[Image.Image], title: str, cols: int = 4) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (400, 30), "white")
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    s = Image.new("RGB", (cols * (w + 4), rows * (h + 4) + 22), "white")
    ImageDraw.Draw(s).text((4, 4), title, fill="black")
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * (w + 4), 22 + (i // cols) * (h + 4)))
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--source", required=True)
    ap.add_argument("--label-orig", required=True)
    ap.add_argument("--pred", required=True, choices=CLASSES)
    ap.add_argument("--fs", default="Lab_hist")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--n-compare", type=int, default=8, help="จำนวนภาพกลุ่มเดียวกันที่ทายถูก ไว้เทียบ")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    cfg = SegConfig()
    rows = load_rows(root)  # trainval เท่านั้น
    attach_boxes(root, rows)
    feats, _ = extract_features(root, rows, cfg, args.workers)
    rows = [r for r in rows if not feats[row_key(r)]["err"]]
    flash = agtron_flash(root)
    for r in rows:
        r["flash"] = flash.get(Path(r["path"]).name, "?")

    Xt = np.array([feats[row_key(r)]["x_train"] for r in rows])
    Xp = np.array([feats[row_key(r)]["x"] for r in rows])
    y = np.array([r["label"] for r in rows])
    src = np.array([r["source"] for r in rows])
    te = src == args.source
    spec = fit_b1(Xt[~te], y[~te], args.fs, args.C)  # เหมือน LOSO fold นี้
    model = LinearSoftmax(spec)
    P = model.proba(Xp[te])
    order = [model.classes.index(c) for c in CLASSES]
    P = P[:, order]
    pred = np.array(CLASSES)[P.argmax(1)]
    held = [r for r, m in zip(rows, te) if m]

    grp = [i for i, r in enumerate(held) if r["label_orig"] == args.label_orig]
    wrong = [i for i in grp if pred[i] == args.pred]
    right = [i for i in grp if pred[i] == held[i]["label"]]
    rng = np.random.default_rng(20261006)
    comp = list(rng.choice(right, size=min(args.n_compare, len(right)), replace=False)) if right else []

    name = f"{args.source}_{args.label_orig}_as_{args.pred}"
    tiles = [tile(root, held[i], cfg, P[i], Xp[te][i]) for i in wrong]
    tiles_c = [tile(root, held[i], cfg, P[i], Xp[te][i]) for i in comp]
    out_dir = root / CONTACT_SHEETS / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet(tiles, f"{args.label_orig} predicted {args.pred} (n={len(wrong)}) - LOSO {args.fs} C={args.C}").save(
        out_dir / f"{name}.jpg", quality=88)
    sheet(tiles_c, f"{args.label_orig} predicted correctly ({held[0]['label'] if held else ''}) - sample {len(comp)}").save(
        out_dir / f"{name}_compare_correct.jpg", quality=88)

    def stats(idx):
        X = Xp[te][idx] if idx else np.zeros((0, len(FEATURES_ALL)))
        g = lambda n: [round(float(v), 2) for v in np.percentile(X[:, FEATURES_ALL.index(n)], [25, 50, 75])] if len(X) else []  # noqa: E731
        return {"n": len(idx), "device": dict(Counter(held[i]["device"] for i in idx)),
                "flash": dict(Counter(held[i]["flash"] for i in idx)),
                "L_med_p25_50_75": g("L_med"), "a_med": g("a_med"), "b_med": g("b_med"), "L_std": g("L_std"),
                "prob_mean": {c: round(float(P[idx, k].mean()), 3) for k, c in enumerate(CLASSES)} if idx else {}}

    # เทียบกับ train ของ fold นี้: ภาพ "light" จาก source อื่นมี L*/b* เท่าไร
    tr_light = (~te) & (y == args.pred)
    ref = {s: [round(float(v), 2) for v in np.percentile(Xt[tr_light & (src == s), FEATURES_ALL.index("L_med")], [25, 50, 75])]
           for s in sorted(set(src[tr_light]))}
    report = {"source": args.source, "label_orig": args.label_orig, "pred": args.pred,
              "model": {"fs": args.fs, "C": args.C, "trained_on": sorted(set(src[~te]))},
              "group_n": len(grp), "pred_counts": dict(Counter(pred[grp])),
              "wrong": stats(wrong), "correct": stats(right),
              f"train_{args.pred}_L_med_p25_50_75_by_source": ref,
              "note": "ดูอย่างเดียว — ห้ามใช้จูน (ml-spec)"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"error_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(root / CACHE / f"error_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "device", "flash", "pred"] + [f"p_{c}" for c in CLASSES])
        for i in wrong:
            w.writerow([held[i]["path"], held[i]["device"], held[i]["flash"], pred[i]] + [round(float(v), 4) for v in P[i]])
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"→ {out_dir / (name + '.jpg')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
