"""เลือกวิธีให้ label รายเมล็ด (บน dev LOSO เท่านั้น) จาก feature ต่อกรอบที่ tools.eval_det_beans เก็บไว้ — ไม่มีการเทรน

    python -m tools.select_bean_labels --eval results/yolo_bean_20261010/eval_holdout_boos_v2 --b1-folds <dir>

ผู้สมัคร (ประกาศไว้ตรงนี้ก่อนรัน): พิกเซลในกรอบ {rect หด 0.2, ellipse หด 0.1, mask = พิกเซลของ segment ∩ กรอบ}
                                    × น้ำหนัก prior ระดับภาพ {0 = per_box, 0.5, 1, 2}
อ้างอิง: "broadcast" = ทุกเมล็ดได้ label ระดับภาพของ B1 (น้ำหนัก B1 ของแต่ละ fold ไม่เคยเห็น source นั้น)
กฎ (ตาม prereg_perbean_v2):
  1. eligible = macro-F1 ราย fold ≥ broadcast − 0.02 ทุก fold และ light↔dark รวม ≤ 0.05 และ error ภาพปน (rf_boos Mixed) < broadcast
  2. ในกลุ่ม eligible เลือก error ภาพปนต่ำสุด · เสมอ → ตามลำดับที่ประกาศ (rect, ellipse, mask · prior น้อยก่อน)
  3. ไม่มีตัว eligible → เลือกตัวที่ error ภาพปน < broadcast และ "ขาดดุล F1 ของ fold ที่แย่สุด" น้อยที่สุด แล้วระบุว่าไม่ผ่านเกณฑ์
นี่คือการเลือกบน dev: ตัวเลขของตัวที่ถูกเลือกมีอคติด้านบวก · ค่าที่ไม่มีอคติต้องมาจาก frozen/test ซึ่งยังไม่ได้รัน
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roastml.linear_model import LinearSoftmax
from roastml.paths import data_dir
from roastml.perbean import fuse_image_prior
from tools.index_sources import read_yolo_names
from tools.perbean_protocol import FOLDS, GATES, dump
from tools.train_perbean import CLASSES, weighted_metrics

SHAPES = {"rect0.2": ("x", {"box_shape": "rect", "box_shrink": 0.2}), "ellipse0.1": ("e", {"box_shape": "ellipse", "box_shrink": 0.1}),
          "mask": ("m", {"box_shape": "mask", "box_shrink": 0.2})}
WEIGHTS = (0.0, 0.5, 1.0, 2.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--b1-folds", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.eval / "label_selection.json"
    if out.exists():
        raise FileExistsError(f"มีผลการเลือกอยู่แล้ว: {out}")
    root = data_dir()
    z = np.load(root / "cache" / f"det_eval_{args.eval.parent.name}_{args.eval.name}.npz", allow_pickle=False)
    per = {r["path"]: r for r in json.loads((args.eval / "per_image.json").read_text(encoding="utf-8"))}
    recs = [{**per[p], "x": z[f"x{j}"], "e": z[f"e{j}"], "m": z[f"m{j}"], "im": z[f"im{j}"]} for j, p in enumerate(list(z["paths"]))]
    models = {h: LinearSoftmax.load(args.b1_folds / h / "model.json") for h in FOLDS}
    if any(list(m.classes) != CLASSES for m in models.values()):
        raise ValueError("ลำดับคลาสของโมเดลราย fold ไม่ตรง")
    names = read_yolo_names(root / "raw/rf_boos/data.yaml")
    mapping = [next(c for c in CLASSES if c in n.lower()) for n in names]

    def labels(model, r, key, w):
        p_img = model.proba(r["im"])[0]
        if w is None:
            return np.full(r["n"], int(p_img.argmax()))
        return fuse_image_prior(model.proba(r[key]), p_img, w).argmax(1)

    cands = {"broadcast": (None, None, {})}
    for sname, (key, cfg) in SHAPES.items():
        for w in WEIGHTS:
            mode = {"bean_label_mode": "per_box"} if w == 0 else {"bean_label_mode": "image_prior", "prior_weight": w}
            cands[f"{sname}|prior{w:g}"] = (key, w, cfg | mode)
    table = {}
    for name, (key, w, card) in cands.items():
        folds, pooled = {}, np.zeros((3, 3))
        for hold in FOLDS:
            yt, ww, pred = [], [], []
            for r in recs:
                if r["source"] != hold or r["label"] not in CLASSES or not r["n"]:
                    continue
                yt += [r["label"]] * r["n"]; ww += [1 / r["n"]] * r["n"]
                pred += [CLASSES[j] for j in labels(models[hold], r, key, w)]
            m = weighted_metrics(yt, pred, ww)
            folds[hold] = m["macro_f1"]
            pooled += np.array(m["cm"])
        mixed = []
        for r in recs:
            if r["source"] == "rf_boos" and r["label"] == "mixed":
                gt = np.bincount([CLASSES.index(mapping[c]) for c in r["gt_classes"]], minlength=3)
                pred = np.bincount(labels(models["rf_boos"], r, key, w), minlength=3) if r["n"] else np.zeros(3, int)
                mixed.append(int(np.abs(pred - gt).sum()))
        table[name] = {"card": card, "fold_macro_f1": folds, "mixed_error_per_image": float(np.mean(mixed)), "mixed_images": len(mixed),
                       "light_dark_rate": float((pooled[0, 2] + pooled[2, 0]) / (pooled[0].sum() + pooled[2].sum()))}
    base = table["broadcast"]
    for name, t in table.items():
        t["fold_delta"] = {h: t["fold_macro_f1"][h] - base["fold_macro_f1"][h] for h in FOLDS}
        t["worst_fold_deficit"] = float(max(0.0, -min(t["fold_delta"].values())))
        t["gates"] = {"noninferior_all_folds": all(d >= -GATES["fold_f1_drop"] for d in t["fold_delta"].values()),
                      "light_dark": t["light_dark_rate"] <= GATES["cross_extreme_rate"],
                      "mixed_better_than_broadcast": t["mixed_error_per_image"] < base["mixed_error_per_image"]}
        t["eligible"] = name != "broadcast" and all(t["gates"].values())
    order = [n for n in table if n != "broadcast"]  # ลำดับประกาศ: rect ก่อน, prior น้อยก่อน
    eligible = [n for n in order if table[n]["eligible"]]
    if eligible:
        chosen = min(eligible, key=lambda n: (round(table[n]["mixed_error_per_image"], 6), order.index(n)))
        status = "eligible on dev gates measured here"
    else:
        pool = [n for n in order if table[n]["gates"]["mixed_better_than_broadcast"]] or order
        chosen = min(pool, key=lambda n: (round(table[n]["worst_fold_deficit"], 6), order.index(n)))
        status = "NO candidate passes all dev gates; chosen by rule 3 (smallest worst-fold deficit among mixed-improving candidates)"
    dump(out, {"rule": __doc__.split("กฎ")[1].split("นี่คือ")[0].strip(), "gates": {k: GATES[k] for k in ("fold_f1_drop", "cross_extreme_rate")},
               "candidates": table, "chosen": chosen, "chosen_card": table[chosen]["card"], "status": status,
               "selection_bias": "chosen on dev LOSO; frozen/test not evaluated"})
    print(f"{'candidate':22s} " + " ".join(f"{h[:9]:>9s}" for h in FOLDS) + "    L<->D  mixed  eligible")
    for n, t in table.items():
        print(f"{n:22s} " + " ".join(f"{t['fold_macro_f1'][h]:9.4f}" for h in FOLDS) + f"  {t['light_dark_rate']:.4f}  {t['mixed_error_per_image']:5.2f}  {t['eligible']}")
    print("chosen:", chosen, "|", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
