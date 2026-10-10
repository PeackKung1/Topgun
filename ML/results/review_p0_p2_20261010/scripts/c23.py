"""C2/C3/C5 checks. Read-only on data; overlays go to $ROAST_DATA_DIR/contact_sheets/review_p0_p2_20261010/."""
import sys, os, json, io, time, random, collections
sys.path.insert(0, ".")
import numpy as np, cv2
from PIL import Image
from roastml.counter import CountConfig, count_beans
from roastml.segment import SegConfig
from roastml.decode import decode_image
from roastml.api import load
from tools.perbean_protocol import DevData
from tools.bench_perbean import jpeg

OUT = sys.argv[1]
SW = "../../roast-classification-seed-count-91d3d9/ML/"
data = DevData(os.environ["ROAST_DATA_DIR"])
sheet_dir = data.root / "contact_sheets" / "review_p0_p2_20261010"
sheet_dir.mkdir(parents=True, exist_ok=True)
cfg = CountConfig.from_dict(json.load(open(SW + "results/counter_final_grid_20261010/selected_config.json", encoding="utf-8")))
seg = SegConfig()
out = {}


def overlay(rgb, res, max_side=420):
    h, w = res.labels.shape
    work = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).copy()
    lab = res.labels
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    work[edge & ((lab > 0) | np.roll(lab, 1, 0).astype(bool) | np.roll(lab, 1, 1).astype(bool))] = (255, 0, 255)
    s = max_side / max(h, w)
    return cv2.resize(work, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def sheet(tiles, cols, path, cell=(430, 330)):
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * cell[1], cols * cell[0], 3), 255, np.uint8)
    for i, (img, text) in enumerate(tiles):
        y, x = (i // cols) * cell[1], (i % cols) * cell[0]
        img = img[:cell[1] - 24, :cell[0] - 6]
        canvas[y + 22:y + 22 + img.shape[0], x + 3:x + 3 + img.shape[1]] = img
        cv2.putText(canvas, text, (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------- C2: Empty report-half failures
per = json.load(open(SW + "results/counter_final_report_20261010/per_image.json", encoding="utf-8"))
emp = [r for r in per if r["source"] == "rf_hendi"]
fails = sorted([r for r in emp if r["n"] > 0], key=lambda r: r["path"])
oks = sorted([r for r in emp if r["n"] == 0], key=lambda r: r["path"])
out["empty_report_half"] = {"n": len(emp), "fail": len(fails), "zero_rate": round(1 - len(fails) / len(emp), 4),
                            "fail_groups": dict(collections.Counter(r["group"] for r in fails)),
                            "all_groups": dict(collections.Counter(r["group"] for r in emp)),
                            "fail_n_hist": dict(collections.Counter(r["n"] for r in fails)),
                            "fail_scene": dict(collections.Counter(r["scene"] for r in fails)),
                            "fail_notes": dict(collections.Counter(tuple(r["notes"]) .__str__() for r in fails))}
tiles, rowsout = [], []
for r in fails:
    row = data.by_path[r["path"]]
    data.assert_safe(row)
    rgb = np.asarray(decode_image((data.root / r["path"]).read_bytes()).image)
    res = count_beans(rgb, cfg, seg)
    h, w = res.labels.shape
    regs = []
    for i in range(1, res.n + 1):
        m = res.labels == i
        ys, xs = np.nonzero(m)
        L, a, b = res.lab[m].mean(0)
        regs.append({"area_frac": round(float(m.mean()), 4), "touch_border": bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2),
                     "cx": round(float(xs.mean() / w), 2), "cy": round(float(ys.mean() / h), 2), "L": round(float(L), 1), "chroma": round(float(np.hypot(a, b)), 1),
                     "hue": round(float(np.degrees(np.arctan2(b, a))), 0)})
    rowsout.append({"path": r["path"], "group": r["group"], "n_logged": r["n"], "n_rerun": res.n, "scene": res.scene, "method": res.method, "notes": res.notes,
                    "wb_applied": bool(res.wb_applied), "regions": regs[:6]})
    tiles.append((overlay(rgb, res), f"n={res.n} {res.scene} {os.path.basename(r['path'])[:28]}"))
for r in oks[:2]:
    rgb = np.asarray(decode_image((data.root / r["path"]).read_bytes()).image)
    tiles.append((overlay(rgb, count_beans(rgb, cfg, seg)), "OK n=0 " + os.path.basename(r["path"])[:28]))
sheet(tiles, 4, sheet_dir / "C2_empty_fail_overlay.jpg")
out["empty_fail_rows"] = rowsout

# ---------- C3: synthetic pile
image = np.full((600, 900, 3), 235, np.uint8)
gt = np.zeros((600, 900), np.int32)
full = []
for i in range(300):
    mask = np.zeros(gt.shape, np.uint8)
    center = (28 + (i % 20) * 44, 25 + (i // 20) * 39)
    cv2.ellipse(mask, center, (27, 23), 0, 0, 360, 1, -1)
    full.append(int(mask.sum()))
    color = ((170, 125, 85), (110, 72, 45), (50, 32, 22))[i % 3]
    image[mask > 0] = color
    gt[mask > 0] = i + 1
    cv2.line(image, (center[0], center[1] - 12), (center[0], center[1] + 12), tuple(max(0, c - 12) for c in color), 2)
vis_area = np.bincount(gt.ravel(), minlength=301)[1:]
out["pile_fixture"] = {"size": [900, 600], "ellipse_axes_px": [27, 23], "grid_pitch_px": [44, 39], "full_area_px": int(np.median(full)),
                       "visible_area_px_median": int(np.median(vis_area)), "visible_frac_min": round(float((vis_area / np.array(full)).min()), 3),
                       "bg_fraction": round(float((gt == 0).mean()), 4), "flat_colors_no_texture": True, "crease_line": "2px, 24px long, colour-12"}


def rejpeg(rgb, q):
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


def iou_pairs(b, t=0.5):
    b = np.asarray(b, float)
    if len(b) < 2:
        return 0
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0), 0, None); ih = np.clip(np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0), 0, None)
    inter = iw * ih; a = b[:, 2] * b[:, 3]
    return int(np.triu(inter / np.maximum(a[:, None] + a - inter, 1e-9) > t, 1).sum())


variants = {"raw_array": image, "jpeg_q85": rejpeg(image, 85), "jpeg_q85_then_q84(browser-like)": rejpeg(rejpeg(image, 85), 84), "jpeg_q60": rejpeg(image, 60)}
out["pile_runs"] = {}
ptiles = []
for name, rgb in variants.items():
    res = count_beans(rgb, cfg, seg)
    h, w = res.labels.shape
    g = cv2.resize(gt.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    owner = np.zeros(res.n + 1, np.int64); purity = np.zeros(res.n + 1)
    areas = np.bincount(res.labels.ravel(), minlength=res.n + 1)
    for i in range(1, res.n + 1):
        v = g[res.labels == i]; bc = np.bincount(v, minlength=301)
        owner[i] = bc.argmax(); purity[i] = bc.max() / max(len(v), 1)
    per_gt = np.bincount(owner[1:][owner[1:] > 0], minlength=301)[1:]
    gt_vis_work = np.bincount(g.ravel(), minlength=301)[1:]
    out["pile_runs"][name] = {"n": res.n, "scene": res.scene, "method": res.method, "notes": res.notes, "work_size": [w, h], "ref_area_work_px": round(res.ref_area, 1),
                              "gt_visible_area_work_px_median": float(np.median(gt_vis_work)),
                              "pred_region_area_median": float(np.median(areas[1:])) if res.n else None,
                              "pred_regions_per_gt_bean": {"mean": round(float(per_gt.mean()), 2), "hist": {str(k): int(v) for k, v in sorted(collections.Counter(per_gt.tolist()).items())}},
                              "pred_regions_on_background": int((owner[1:] == 0).sum()),
                              "pred_regions_with_purity<0.8(span >1 bean)": int((purity[1:] < 0.8).sum()),
                              "gt_beans_with_0_regions": int((per_gt == 0).sum()),
                              "dup_box_pairs_iou>0.5": iou_pairs(res.bboxes), "labels_are_disjoint_partition": True}
    ptiles.append((overlay(rgb, res, 880), f"{name}: n={res.n} (visible GT 300)"))
sheet(ptiles, 2, sheet_dir / "C3_synthetic_pile_overlay.jpg", cell=(890, 620))
# crop for close look
res = count_beans(variants["jpeg_q85"], cfg, seg)
h, w = res.labels.shape
ov = overlay(variants["jpeg_q85"], res, max(h, w))
cv2.imwrite(str(sheet_dir / "C3_pile_crop_zoom.png"), cv2.cvtColor(cv2.resize(ov[100:300, 200:500], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))

# ---------- pile reality check: agtron full frame (FW path) vs ROI+force_pile (P1 eval path)
pred = load(SW + "models/perbean_v3_preview_20261010")
rng = random.Random(20261010)
ag = rng.sample(sorted([r for r in data.rows if r["source"] == "agtron" and r["label"] in ("light", "medium", "dark")], key=lambda r: r["path"]), 20)
rows = []
atiles = []
for r in ag:
    raw = (data.root / r["path"]).read_bytes()
    a = pred.predict_bytes(raw); b = pred.predict_dataset(raw, source="agtron", roi=r["roi"])
    rows.append({"path": os.path.basename(r["path"]), "true": r["label"], "fw_n": a["n_beans"], "fw_label": a["label"], "fw_method": a["count_method"], "fw_warn": a["warnings"],
                 "fw_ms": a["timing_ms"]["total"], "roi_n": b["n_beans"], "roi_label": b["label"], "image_size": a["image_size"]})
    if len(atiles) < 4:
        rgb = np.asarray(decode_image(raw).image)
        atiles.append((overlay(rgb, count_beans(rgb, cfg, seg)), f"FW path n={a['n_beans']} {a['count_method']} | ROI path n={b['n_beans']}"))
sheet(atiles, 2, sheet_dir / "C4_agtron_fullframe_overlay.jpg")
out["agtron_fw_vs_roi"] = {"n": len(rows), "fw_n_beans": sorted(r["fw_n"] for r in rows), "roi_n_beans": sorted(r["roi_n"] for r in rows),
                           "fw_label_right": sum(r["fw_label"] == r["true"] for r in rows), "roi_label_right": sum(r["roi_label"] == r["true"] for r in rows),
                           "fw_ms_p50_max": [float(np.median([r["fw_ms"] for r in rows])), max(r["fw_ms"] for r in rows)], "rows": rows}

# ---------- C5: cap + worst-case loop cost
g = np.random.default_rng(3)
speck = np.full((1200, 1600, 3), 235, np.uint8)
for _ in range(4000):
    x, y = int(g.integers(5, 1595)), int(g.integers(5, 1195))
    cv2.circle(speck, (x, y), int(g.integers(3, 7)), (90, 60, 40), -1)
t = time.perf_counter(); rs = count_beans(speck, cfg, seg); t_speck = (time.perf_counter() - t) * 1000
ok, buf = cv2.imencode(".jpg", cv2.cvtColor(speck, cv2.COLOR_RGB2BGR))
resp = pred.predict_bytes(buf.tobytes())
out["cap_test_4000_dots"] = {"count_ms": round(t_speck, 1), "n": rs.n, "method": rs.method, "notes": rs.notes, "api_n_beans": resp["n_beans"], "api_len_beans": len(resp["beans"]),
                             "api_count_method": resp["count_method"], "api_warnings": resp["warnings"], "api_status": resp["status"], "api_total_ms": resp["timing_ms"]["total"],
                             "api_stage_ms": {k: resp["timing_ms"][k] for k in ("decode", "WB", "segment", "count", "features", "classify", "group")}}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("sheets:", sheet_dir)
