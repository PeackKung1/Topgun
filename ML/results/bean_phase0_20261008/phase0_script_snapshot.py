"""Phase 0 bean-count exploration (trainval only, read + measure). argv: data_dir out_json"""
import csv, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from roastml.decode import decode_image
from roastml.features import FEATURE_SETS, FEATURES_ALL, box_features
from roastml.segment import SegConfig, _odd, background_white_balance, segment, to_work
from tools.index_sources import read_yolo_boxes, read_yolo_names

root = Path(sys.argv[1]); out = Path(sys.argv[2])
cfg = SegConfig()
SRC = ("rf_robusta", "rf_boos")
names = {s: read_yolo_names(root / "raw" / s / "data.yaml") for s in SRC}
CLASSMAP = {"Dark Roast": "dark", "Light Roast": "light", "Medium Roast": "medium", "Maw": "maw", "Raw": "green"}

rows = [r for r in csv.DictReader(open(root / "manifest.csv", encoding="utf-8"))]
dropped = [r for r in csv.DictReader(open(root / "cache/manifest_dropped.csv", encoding="utf-8"))]
report = {"data": {}, "count": {}, "pixels": {}, "features": {}}

# ---------------- 0.1 data
for s in SRC:
    tv = [r for r in rows if r["source"] == s and r["split"] == "trainval"]
    nb, mixed, single, cls_per = [], 0, 0, Counter()
    for r in tv:
        b = read_yolo_boxes(root / r["path"]); nb.append(len(b))
        cl = {CLASSMAP[names[s][c]] for c, *_ in b}
        if len(cl) > 1: mixed += 1
        if len(b) == 1: single += 1
        for c, *_ in b: cls_per[CLASSMAP[names[s][c]]] += 1
    dr = [d for d in dropped if d["source"] == s]
    report["data"][s] = {"trainval_images": len(tv), "labels": dict(Counter(r["label"] for r in tv)),
                         "boxes_median": float(np.median(nb)), "boxes_p90": float(np.percentile(nb, 90)),
                         "boxes_max": int(max(nb)), "images_multi_class_boxes": mixed,
                         "images_single_box": single, "box_class_counts": dict(cls_per),
                         "dropped_from_manifest": dict(Counter(d["reason"] for d in dr)),
                         "test_split_images_not_opened": sum(r["source"] == s and r["split"] == "test" for r in rows)}

# ---------------- counters
def fg_candidates(lab, bg, h, w):
    dist = np.linalg.norm(lab - bg, axis=2)
    t, _ = cv2.threshold(np.clip(dist * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (dist > max(t / 2.0, cfg.min_contrast)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * 0.006),) * 2)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (cfg.min_area_frac <= a / (h * w) <= cfg.max_area_frac): continue
        sol = a / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        cand.append((c, a, sol, max(rw, rh) / max(min(rw, rh), 1.0)))
    return cand

def accepted(cand):
    good = [a for _, a, sol, asp in cand if sol >= cfg.min_solidity and asp <= cfg.max_aspect]
    med = float(np.median(good if good else [a for _, a, _, _ in cand]))
    acc = []
    for c, a, sol, asp in cand:
        if a < cfg.min_rel_area * med: continue
        if a > cfg.cluster_area_ratio * med:
            if sol < cfg.cluster_min_solidity: continue
        elif sol < cfg.min_solidity or asp > cfg.max_aspect: continue
        acc.append((c, a, sol, asp))
    return acc, good, med

def watershed_count(acc, med):
    n = 0
    for c, a, sol, asp in acc:
        if a <= 1.8 * med:
            n += 1; continue
        x, y, bw, bh = cv2.boundingRect(c)
        rm = np.zeros((bh + 2, bw + 2), np.uint8)
        cv2.drawContours(rm, [c - [x - 1, y - 1]], -1, 255, cv2.FILLED)
        dt = cv2.distanceTransform(rm, cv2.DIST_L2, 5)
        r = max(2.0, 0.5 * np.sqrt(med / np.pi))           # half the single-bean radius
        peaks = (dt >= 0.6 * r).astype(np.uint8)
        # local maxima: dilate compare
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(r * 1.2),) * 2)
        lm = ((dt == cv2.dilate(dt, k)) & (dt >= 0.6 * r)).astype(np.uint8)
        nlab, _ = cv2.connectedComponents(lm)
        n += max(1, nlab - 1)
    return n

def area_count(acc, good, med):
    hi = [a for _, a, sol, asp in acc if sol >= 0.9 and asp <= cfg.max_aspect and a <= 1.8 * med]
    unit = float(np.median(hi)) if hi else med
    return int(round(sum(a for _, a, _, _ in acc) / max(unit, 1.0)))

def bucket(n):
    return "1" if n == 1 else "2-5" if n <= 5 else "6-15" if n <= 15 else "16+"

recs = []
px_per_bean = defaultdict(list)
feat_rows = []
for s in SRC:
    tv = [r for r in rows if r["source"] == s and r["split"] == "trainval"]
    for r in tv:
        boxes = read_yolo_boxes(root / r["path"])
        if not boxes: continue
        d = decode_image((root / r["path"]).read_bytes())
        rgb = np.asarray(d.image)
        t0 = time.perf_counter(); seg = segment(rgb, cfg); t_seg = (time.perf_counter() - t0) * 1000
        work, sc = to_work(rgb, cfg); h, w = work.shape[:2]
        rec = {"source": s, "label": r["label"], "n_gt": len(boxes), "mode": seg.mode, "seg_n": seg.n_regions,
               "t_seg": t_seg}
        if seg.mode == "beans":
            t0 = time.perf_counter()
            wbr = background_white_balance(work.astype(np.float32) / 255, cfg, allow=True)
            cand = fg_candidates(wbr.lab, wbr.bg, h, w); acc, good, med = accepted(cand)
            t_base = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter(); rec["ws_n"] = watershed_count(acc, med); rec["t_ws"] = (time.perf_counter() - t1) * 1000
            t1 = time.perf_counter(); rec["area_n"] = area_count(acc, good, med); rec["t_area"] = (time.perf_counter() - t1) * 1000
            rec["t_base"] = t_base
            rec["merged_blobs"] = sum(a > 1.8 * med for _, a, _, _ in acc); rec["n_blobs"] = len(acc)
        recs.append(rec)
        for c, cx, cy, bw, bh in boxes:
            px_per_bean[s].append(bw * w * bh * h * np.pi / 4)
        # 0.4 per-bean vs whole-image Lab_hist features (3-class images only, cap 3 beans/image)
        if r["label"] in ("light", "medium", "dark") and len(feat_rows) < 6000:
            img_x = box_features(rgb, [b[1:] for b in boxes], cfg).x
            for b in boxes[:3]:
                try:
                    bx = box_features(rgb, [b[1:]], cfg).x
                    feat_rows.append({"source": s, "label": r["label"], "bean_cls": CLASSMAP[names[s][b[0]]],
                                      "img": img_x.tolist(), "bean": bx.tolist()})
                except ValueError:
                    pass

def summarize(sel, key):
    if not sel: return None
    gt = np.array([x["n_gt"] for x in sel], float); pr = np.array([x[key] for x in sel], float)
    err = np.abs(pr - gt); rel = err / gt
    return {"n": len(sel), "MAE": float(err.mean()), "median_rel_err": float(np.median(rel)),
            "within_20pct_or_2": float(np.mean((rel <= 0.2) | (err <= 2)))}

cnt = {}
for s in SRC + ("ALL",):
    S_ = [x for x in recs if s == "ALL" or x["source"] == s]
    cnt[s] = {"modes": dict(Counter(x["mode"] for x in S_))}
    for b in ("1", "2-5", "6-15", "16+", "all"):
        sb = [x for x in S_ if b == "all" or bucket(x["n_gt"]) == b]
        beans = [x for x in sb if x["mode"] == "beans"]
        cnt[s][b] = {"n_images": len(sb), "beans_mode": len(beans),
                     "a_contour(all modes, non-beans=0)": summarize(sb, "seg_n"),
                     "a_contour(beans mode)": summarize(beans, "seg_n"),
                     "b_watershed(beans mode)": summarize(beans, "ws_n"),
                     "c_area(beans mode)": summarize(beans, "area_n")}
    beans = [x for x in S_ if x["mode"] == "beans"]
    if beans:
        cnt[s]["merged_blob_rate"] = float(sum(x["merged_blobs"] for x in beans) / max(1, sum(x["n_blobs"] for x in beans)))
        cnt[s]["images_with_merged_blob"] = float(np.mean([x["merged_blobs"] > 0 for x in beans]))
        for k in ("t_seg", "t_base", "t_ws", "t_area"):
            v = [x[k] for x in (beans if k != "t_seg" else S_)]
            cnt[s][f"{k}_ms_p50_p95"] = [float(np.percentile(v, 50)), float(np.percentile(v, 95))]
report["count"] = cnt
report["pixels"] = {s: {"bean_px_at_work_median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                        "frac_below_150": float(np.mean(np.array(v) < 150))} for s, v in px_per_bean.items()}

# 0.4 feature comparison
idx = [FEATURES_ALL.index(n) for n in FEATURE_SETS["Lab_hist"]]
I = np.array([f["img"] for f in feat_rows])[:, idx]; B = np.array([f["bean"] for f in feat_rows])[:, idx]
sd = I.std(0) + 1e-9
L50 = FEATURES_ALL.index("L_p50") if "L_p50" in FEATURES_ALL else 0
fc = {"n_beans": len(feat_rows), "mean_abs_z_diff_bean_vs_image": float(np.mean(np.abs(B - I) / sd)),
      "per_feature_mean_abs_z": {FEATURE_SETS["Lab_hist"][i]: float(np.mean(np.abs(B[:, i] - I[:, i]) / sd[i])) for i in range(len(idx))}}
# L* median per bean by class/source (descriptive, within-source ordering only)
Lb = defaultdict(list)
for f in feat_rows:
    Lb[(f["source"], f["bean_cls"])].append(f["bean"][0])
fc["bean_L_p50_median_by_source_class"] = {f"{s}:{c}": float(np.median(v)) for (s, c), v in sorted(Lb.items())}
report["features"] = fc
report["feature_name_index0"] = FEATURES_ALL[0]
json.dump(report, open(out, "w", encoding="utf-8"), indent=1)
print("done", len(recs))
