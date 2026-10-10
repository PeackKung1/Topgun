"""D3/D4 diagnosis from the P1 feature cache + saved fold models. No fitting, no selection."""
import json, sys, collections
import numpy as np
sys.path.insert(0, ".")
from roastml.linear_model import LinearSoftmax
from roastml.features import FEATURES_ALL

OUT = sys.argv[1]
D = "C:/Users/Public/Documents/TOPGUN_CONTEST/data/"
SW = "../../roast-classification-seed-count-91d3d9/ML/"
meta = json.load(open(D + "cache/perbean_features_d804b44428eb274a.json", encoding="utf-8"))
z = np.load(D + "cache/perbean_features_d804b44428eb274a.npz", allow_pickle=False)
recs = [{**r, "X": z[f"x{i}"], "image_x": z[f"im{i}"], "train_x": z[f"tr{i}"]} for i, r in enumerate(meta["records"])]
CL = ["light", "medium", "dark"]
FOLDS = ["ontoum224", "rf_robusta", "rf_boos", "agtron"]
iL, ia, ib, iS, iP10 = (FEATURES_ALL.index(k) for k in ("L_med", "a_med", "b_med", "L_std", "L_p10"))
single = [r for r in recs if r["row"]["label"] in CL and r["n"]]
out = {}


def f1(y, p, w):
    cm = np.zeros((3, 3))
    for a, b, c in zip(y, p, w):
        cm[a, b] += c
    f = [2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum()) if cm[i].sum() + cm[:, i].sum() else 0.0 for i in range(3)]
    return round(float(np.mean(f)), 4), [round(x, 3) for x in f]


# 1) feature table per source x label: per-bean (image mean of beans) vs B1 pipeline view vs B1 train view
tab = []
for s in FOLDS:
    for l in CL:
        rr = [r for r in single if r["row"]["source"] == s and r["row"]["label"] == l]
        if not rr:
            tab.append({"source": s, "label": l, "images": 0})
            continue
        xb = np.array([r["X"].mean(0) for r in rr]); xi = np.array([r["image_x"] for r in rr]); xt = np.array([r["train_x"] for r in rr])
        tab.append({"source": s, "label": l, "images": len(rr), "beans": int(sum(r["n"] for r in rr)),
                    "bean_L_med": round(float(xb[:, iL].mean()), 1), "img_L_med": round(float(xi[:, iL].mean()), 1), "b1train_L_med": round(float(xt[:, iL].mean()), 1),
                    "bean_a": round(float(xb[:, ia].mean()), 1), "img_a": round(float(xi[:, ia].mean()), 1),
                    "bean_b": round(float(xb[:, ib].mean()), 1), "img_b": round(float(xi[:, ib].mean()), 1),
                    "bean_L_std": round(float(xb[:, iS].mean()), 1), "img_L_std": round(float(xi[:, iS].mean()), 1),
                    "bean_L_p10": round(float(xb[:, iP10].mean()), 1), "img_L_p10": round(float(xi[:, iP10].mean()), 1)})
out["feature_table"] = tab

# 2) ontoum: per-bean vs whole-image feature of the same image, 50 seeded images + all 895
on = sorted([r for r in single if r["row"]["source"] == "ontoum224"], key=lambda r: r["row"]["path"])
rng = np.random.default_rng(20261010)
pick = [on[i] for i in sorted(rng.choice(len(on), 50, replace=False))]


def diffstats(rr):
    d = np.array([r["X"][0] - r["image_x"] for r in rr])
    return {k: {"mean_bean": round(float(np.mean([r["X"][0][i] for r in rr])), 2), "mean_img": round(float(np.mean([r["image_x"][i] for r in rr])), 2),
                "mean_diff": round(float(d[:, i].mean()), 2), "sd_diff": round(float(d[:, i].std()), 2), "max_abs_diff": round(float(np.abs(d[:, i]).max()), 2)}
            for k, i in (("L_med", iL), ("a_med", ia), ("b_med", ib), ("L_p10", iP10), ("L_std", iS))}


out["ontoum_bean_vs_image_50"] = {"n": 50, "n_beans_all_1": all(r["n"] == 1 for r in pick), "by_label": {l: diffstats([r for r in pick if r["row"]["label"] == l]) for l in CL}, "all": diffstats(pick)}
out["ontoum_bean_vs_image_895"] = diffstats(on)

# 3) 2x2 swap: which model x which feature view (existing fold models only)
swap = {}
for hold in FOLDS:
    md = SW + f"models/perbean_final_loso_20261010/{hold}/"
    b1 = LinearSoftmax.load(md + "model.json"); bean = LinearSoftmax.load(md + "bean_model.json")
    test = [r for r in single if r["row"]["source"] == hold]
    y, w, XB, XI = [], [], [], []
    for r in test:
        n = r["n"]; y += [CL.index(r["row"]["label"])] * n; w += [1 / n] * n; XB.append(r["X"]); XI += [r["image_x"]] * n
    XB = np.concatenate(XB); XI = np.array(XI)
    swap[hold] = {"B1model_x_imageFeat(ref)": f1(y, b1.proba(XI).argmax(1), w), "B1model_x_beanFeat": f1(y, b1.proba(XB).argmax(1), w),
                  "beanModel_x_imageFeat": f1(y, bean.proba(XI).argmax(1), w), "beanModel_x_beanFeat(P1)": f1(y, bean.proba(XB).argmax(1), w),
                  "bean_model_C": None, "bean_W_norm": round(float(np.linalg.norm(bean.W)), 2), "b1_W_norm": round(float(np.linalg.norm(b1.W)), 2)}
out["swap_2x2_macroF1_and_perclass[L,M,D]"] = swap

# 4) training weight share per source per class (bean model: 1/n per image, no video cap) vs B1 broadcast rows (10-frame video cap)
share = {}
for hold in FOLDS:
    tr = [r for r in single if r["row"]["source"] != hold]
    wt = collections.Counter(); b1rows = collections.Counter(); groups = collections.defaultdict(list)
    for r in tr:
        wt[(r["row"]["label"], r["row"]["source"])] += 1.0
        groups[r["row"]["group"]].append(r)
    for g, rr in groups.items():
        k = min(len(rr), 10) if ":video:" in g else len(rr)
        b1rows[(rr[0]["row"]["label"], rr[0]["row"]["source"])] += k
    share[hold] = {}
    for l in CL:
        tot = sum(v for (ll, s), v in wt.items() if ll == l); tot1 = sum(v for (ll, s), v in b1rows.items() if ll == l)
        share[hold][l] = {"bean_model_image_weight": int(tot), "bean_share_by_source": {s: round(wt[(l, s)] / tot, 3) for s in FOLDS if wt[(l, s)]},
                          "b1_rows(video cap 10)": int(tot1), "b1_share_by_source": {s: round(b1rows[(l, s)] / tot1, 3) for s in FOLDS if b1rows[(l, s)]}}
    share[hold]["n_video_groups_boos"] = len({g for g in groups if g.startswith("rf_boos:video")})
out["train_weight_share"] = share
# zero-detection images excluded from training (b1 broadcast uses them: fit_broadcast gets all train records)
out["note_b1_fit_rows_include_zero_detection_images"] = sum(1 for r in recs if r["row"]["label"] in CL and r["n"] == 0)

# 5) D4: light<->dark of B-bean by fold and direction (image-weighted)
ld = {}
for hold in FOLDS:
    bean = LinearSoftmax.load(SW + f"models/perbean_final_loso_20261010/{hold}/bean_model.json")
    cm = np.zeros((3, 3))
    for r in [r for r in single if r["row"]["source"] == hold]:
        p = bean.proba(r["X"]).argmax(1)
        for j in p:
            cm[CL.index(r["row"]["label"]), j] += 1 / r["n"]
    ld[hold] = {"light->dark": round(cm[0, 2], 1), "dark->light": round(cm[2, 0], 1), "light_imgs": round(cm[0].sum(), 1), "dark_imgs": round(cm[2].sum(), 1)}
tot = sum(v["light->dark"] + v["dark->light"] for v in ld.values()); den = sum(v["light_imgs"] + v["dark_imgs"] for v in ld.values())
out["light_dark_by_fold"] = {"folds": ld, "pooled_rate": round(tot / den, 4), "share_of_errors": {h: round((v["light->dark"] + v["dark->light"]) / tot, 3) for h, v in ld.items()}}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
