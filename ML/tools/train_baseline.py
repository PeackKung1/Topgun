"""Baseline B0 / B1-small / B1 + LOSO 5 fold + shortcut probe → ML/results/baseline_loso.{csv,json}
และ export ตัวชนะ → ML/models/current/ (gitignored)

โปรโตคอล (ml-spec ข้อ 5, 7, 8 + results/split_decisions.md):
- ข้อมูล: manifest แถว split=trainval และ label ∈ {light, medium, dark} เท่านั้น · **ไม่โหลด split=test เลย**
- LOSO 4 fold หน่วย = source: ontoum224, rf_robusta, rf_boos, agtron (5 เครื่อง)
  rf_hendi ตัดออกจาก baseline (8 ต.ค.: ทุกภาพอยู่ในเครื่องคั่วเดียว พื้นหลังทายคลาสได้ 0.85, light/medium แยกด้วยสีไม่ได้)
- รูป: อ่าน bytes → roastml.decode.decode_image (เส้นทางเดียวกับ Pi) แล้วสกัด feature 2 view:
    train view    : agtron = ROI · rf_robusta/rf_boos = พิกเซลใน bbox YOLO · ontoum224 = smart crop
    pipeline view : smart-crop pipeline จริงแบบที่ Pi เห็น (agtron ยังเป็น ROI — ห้ามใช้ภาพเต็ม agtron)
  เทรนด้วย train view เสมอ · ประเมิน fold ที่ hold out 2 แบบ: (ก) train view (bbox)  (ข) pipeline view
- B0: threshold 2 ค่าบน L_med · B1-small: median L*/a*/b*, IQR L*, chroma + LogReg C เล็ก · B1: Lab_hist + LogReg
- เลือกตัวที่ดีที่สุดของแต่ละโมเดลด้วย mean LOSO macro-F1 ของ (ข) pipeline view (= สิ่งที่ deploy เห็นจริง)
  ปัด 2 ตำแหน่ง → โมเดลง่ายกว่า → C เล็กกว่า · ตัวเลขของ config ที่ถูกเลือกจึงมองโลกในแง่ดีเล็กน้อย
- export: --export-model auto (ตามกฎข้างบน) หรือระบุ B0 / B1-small / B1 (ผู้ใช้ตัดสิน 8 ต.ค.: B1)
- shortcut probe: ทายคลาสจาก "ขอบภาพ" อย่างเดียว ภายในแต่ละ source (StratifiedGroupKFold ตาม group)

ใช้: python -m tools.train_baseline [--workers 8] [--export-model B1] [--no-export]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import roastml
from roastml.decode import decode_image
from roastml.features import (
    BORDER_NAMES, FEATURE_SETS, FEATURES_ALL, border_features, box_features, image_features, select,
)
from roastml.linear_model import LinearSoftmax, Threshold3, spec_from_sklearn
from roastml.paths import CACHE, data_dir
from roastml.segment import SegConfig
from tools.index_sources import read_yolo_boxes

ML_DIR = Path(__file__).resolve().parent.parent
RESULTS = ML_DIR / "results"
MODEL_OUT = ML_DIR / "models" / "current"  # gitignored (เทรนรวม agtron license Unknown)
CLASSES = ["light", "medium", "dark"]          # ลำดับใน report
ORDINAL = {"dark": 0, "medium": 1, "light": 2}  # สำหรับ B0 (L* ต่ำ = เข้ม)
FOLDS = ["ontoum224", "rf_robusta", "rf_boos", "agtron"]
# source ที่ไม่ใช้ใน baseline (ยังอยู่ใน manifest) — results/split_decisions.md
EXCLUDED_SOURCES = {"rf_hendi": "ภาพในเครื่องคั่วเดียวทั้งชุด · ขอบภาพทายคลาสได้ (probe 0.85) · light≈medium ใน L*"}
MODELS = ("B0", "B1-small", "B1")
BOX_SOURCES = ("rf_robusta", "rf_boos")  # มี bbox YOLO → train view ใช้พิกเซลใน bbox
# (ชื่อโมเดล, feature set, C grid) · ลำดับ = ความซับซ้อน (ใช้ตัดสินเมื่อคะแนนเท่ากัน)
B1_GRID = [("B1-small", "small", [0.001, 0.01, 0.1]), ("B1", "Lab_hist", [0.01, 0.1, 1.0, 10.0])]
# 0.0 = ตอบ argmax เสมอ status "ok" (ยังไม่ abstain / ไม่ใช้ low_confidence) · confidence ส่งแยกใน field
LOW_CONF_THRESHOLD = 0.0
SEED = 20261006


# ================================================================ metrics (ทดสอบใน tests/test_baseline.py)
def confusion(y: np.ndarray, p: np.ndarray, classes: list[str] = CLASSES) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, p):
        m[idx[a], idx[b]] += 1
    return m


def prf_from_cm(m: np.ndarray, classes: list[str] = CLASSES) -> dict:
    out = {}
    for i, c in enumerate(classes):
        tp = m[i, i]
        prec = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        rec = tp / m[i].sum() if m[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(m[i].sum())}
    return out


def metrics(y: np.ndarray, p: np.ndarray, classes: list[str] = CLASSES) -> dict:
    """acc, macro-F1 (เฉลี่ยเฉพาะคลาสที่มีใน y), P/R/F1 ราย class, อัตราผิดข้ามขั้น light↔dark (ต่อทั้งหมด)"""
    m = confusion(y, p, classes)
    pc = prf_from_cm(m, classes)
    present = [c for c in classes if pc[c]["support"] > 0]
    n = int(m.sum())
    li, di = classes.index("light"), classes.index("dark")
    cross = int(m[li, di] + m[di, li])
    return {
        "n": n,
        "acc": float(np.trace(m) / n) if n else 0.0,
        "macro_f1": float(np.mean([pc[c]["f1"] for c in present])) if present else 0.0,
        "per_class": pc,
        "cross_step": cross,
        "cross_step_rate": cross / n if n else 0.0,
        "cm": m.tolist(),
    }


# ================================================================ B0
def fit_b0(v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """เลือก t1 ≤ t2 บน L_med ให้ macro-F1 สูงสุด (candidate = percentile 1..99 ของ v)"""
    cand = np.unique(np.percentile(v, np.arange(1, 100)))
    yi = np.array([ORDINAL[c] for c in y])
    best, best_t = -1.0, (float(cand[0]), float(cand[-1]))
    for i, t1 in enumerate(cand):
        for t2 in cand[i:]:
            pi = np.where(v < t1, 0, np.where(v < t2, 1, 2))
            f1s = []
            for k in range(3):
                tp = np.sum((pi == k) & (yi == k))
                fp, fn = np.sum((pi == k) & (yi != k)), np.sum((pi != k) & (yi == k))
                f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
            f = float(np.mean(f1s))
            if f > best:
                best, best_t = f, (float(t1), float(t2))
    return best_t


def b0_spec(t: tuple[float, float]) -> dict:
    return {"type": "threshold", "feature": "L_med", "classes": ["dark", "medium", "light"], "thresholds": list(t)}


# ================================================================ B1
def make_pipe(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))])


def fit_b1(X_all: np.ndarray, y: np.ndarray, fs: str, C: float) -> dict:
    names = FEATURE_SETS[fs]
    pipe = make_pipe(C).fit(select(X_all, names), y)
    return spec_from_sklearn(pipe, names, CLASSES)


def predict_spec(spec: dict, X_all: np.ndarray) -> np.ndarray:
    model = Threshold3(spec) if spec["type"] == "threshold" else LinearSoftmax(spec)
    return np.array(model.classes)[model.proba(X_all).argmax(1)]


# ================================================================ features (multiprocess + cache)
def _extract_one(job: tuple[str, str, dict, list | None]) -> dict:
    """x = pipeline view (smart crop / ROI) · x_train = bbox ถ้ามี ไม่งั้น = x · border = ขอบภาพสำหรับ probe"""
    path, roi, cfg, boxes = job
    try:
        img = decode_image(Path(path).read_bytes())
        rgb = np.asarray(img.image)
        if roi:  # agtron: พิกัด ROI อ้างอิงภาพเต็มหลังหมุน EXIF → สเกลตามภาพที่ decode แล้ว
            s = img.scale
            x1, y1, x2, y2 = (int(round(int(v) * s)) for v in roi.split())
            rgb = rgb[y1:y2, x1:x2]
        c = SegConfig.from_dict(cfg)
        f = image_features(rgb, c, find_beans=not roi)
        x_train = box_features(rgb, boxes, c)[0] if boxes else f.x
        return {"x": f.x.tolist(), "x_train": np.asarray(x_train).tolist(), "border": border_features(rgb).tolist(),
                "mode": f.seg.mode, "wb": f.seg.wb_applied, "ms": f.ms, "err": ""}
    except Exception as e:  # รายงานรวมทีหลัง ไม่ให้ทั้งรอบล้ม
        return {"err": f"{type(e).__name__}: {e}"}


def row_key(r: dict) -> str:
    return f"{r['md5']}|{r['roi']}|{'box' if r.get('boxes') else ''}"


def extract_features(root: Path, rows: list[dict], cfg: SegConfig, workers: int) -> tuple[dict, str]:
    cfg_d = cfg.to_dict()
    tag = hashlib.md5(json.dumps({"cfg": cfg_d, "v": roastml.__version__, "feat": FEATURES_ALL,
                                  "border": BORDER_NAMES, "views": 2,
                                  "decode": "draft_aspect"}, sort_keys=True).encode()).hexdigest()[:10]
    cache_path = root / CACHE / f"baseline_features_{tag}.json"
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"cache เสีย ({e}) → สร้างใหม่")
    todo = [r for r in rows if row_key(r) not in cache]
    if todo:
        print(f"สกัด feature {len(todo)} ภาพ (cache {len(rows) - len(todo)}) workers={workers}")
        t0 = time.time()
        jobs = [(str(root / r["path"]), r["roi"], cfg_d, r.get("boxes")) for r in todo]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_extract_one, jobs, chunksize=16))
        else:
            res = [_extract_one(j) for j in jobs]
        for r, e in zip(todo, res):
            cache[row_key(r)] = e
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, cache_path)
        print(f"  เสร็จใน {time.time() - t0:.0f}s → {cache_path.name}")
    return {row_key(r): cache[row_key(r)] for r in rows}, cache_path.name


# ================================================================ LOSO
def loso_predict(X: np.ndarray, y: np.ndarray, src: np.ndarray, fit, X_eval: dict | None = None,
                 folds: list[str] = FOLDS):
    """fit(X_tr, y_tr) → spec ด้วย X (train view) · ทำนายแถวของ source ที่ถูก hold out
    X_eval=None → คืน pred บน X · X_eval={"ชื่อ": matrix} → คืน {"ชื่อ": pred} (fit เดียวกันทุก view)"""
    evals = {"_": X} if X_eval is None else X_eval
    preds = {k: np.empty(len(y), dtype=object) for k in evals}
    for s in folds:
        te = src == s
        if not te.any():
            continue
        spec = fit(X[~te], y[~te])
        for k, Xe in evals.items():
            preds[k][te] = predict_spec(spec, Xe[te])
    preds = {k: v.astype(str) for k, v in preds.items()}
    return preds["_"] if X_eval is None else preds


def fold_table(y, p, src, folds: list[str] = FOLDS) -> dict:
    out = {s: metrics(y[src == s], p[src == s]) for s in folds if (src == s).any()}
    keys = ["acc", "macro_f1", "cross_step_rate"]
    out["mean"] = {k: float(np.mean([out[s][k] for s in folds if s in out])) for k in keys}
    m = np.isin(src, folds)
    out["pooled"] = metrics(y[m], p[m])
    return out


def agtron_breakdown(label_orig: np.ndarray, p: np.ndarray, src: np.ndarray) -> dict:
    m = src == "agtron"
    tab: dict[str, Counter] = defaultdict(Counter)
    for lo, pp in zip(label_orig[m], p[m]):
        tab[lo][pp] += 1
    return {lo: {c: tab[lo].get(c, 0) for c in CLASSES} for lo in sorted(tab, key=lambda s: -int(s.split("_")[1]))}


# ================================================================ shortcut probe
def within_source_cv(F: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1.0) -> dict | None:
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(set(groups))
    if n_groups < 2 or len(set(y)) < 2:
        return None
    k = min(5, n_groups)
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(F, y, groups):
        if len(set(y[tr])) < 2:
            pred[te] = Counter(y[tr]).most_common(1)[0][0]
            continue
        pred[te] = make_pipe(C).fit(F[tr], y[tr]).predict(F[te])
    m = metrics(y, pred.astype(str))
    return {"macro_f1": m["macro_f1"], "acc": m["acc"], "n_splits": k, "n_groups": n_groups}


# ================================================================ main
def load_rows(root: Path) -> list[dict]:
    with open(root / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # กันพลาด: ไม่แตะ split=test แม้แต่อ่าน feature
    return [r for r in rows if r["split"] == "trainval" and r["label"] in CLASSES and r["source"] in FOLDS]


def attach_boxes(root: Path, rows: list[dict]) -> int:
    """ใส่ r["boxes"] (cx, cy, w, h) ให้แถวของ BOX_SOURCES · คืนจำนวนแถวที่ไม่มี bbox (จะใช้ pipeline view แทน)"""
    missing = 0
    for r in rows:
        if r["source"] in BOX_SOURCES:
            r["boxes"] = [list(b[1:]) for b in read_yolo_boxes(root / r["path"])]
            missing += not r["boxes"]
    return missing


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _csv_rows(model: str, config: str, view: str, tab: dict, folds: list[str]) -> list[dict]:
    out = []
    for fold in folds + ["mean", "pooled"]:
        if fold not in tab:
            continue
        t = tab[fold]
        row = {"model": model, "config": config, "eval_view": view, "fold": fold,
               "n": t.get("n", ""), "acc": round(t["acc"], 4), "macro_f1": round(t["macro_f1"], 4),
               "cross_step_rate": round(t["cross_step_rate"], 4)}
        for c in CLASSES:
            pc = t.get("per_class", {}).get(c, {})
            for k, short in (("precision", "p"), ("recall", "r"), ("f1", "f1")):
                row[f"{c}_{short}"] = round(pc[k], 4) if pc else ""
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--export-model", choices=("auto",) + MODELS, default="auto",
                    help="auto = ตามกฎเลือก · ระบุชื่อ = export ตัวที่ดีที่สุดของโมเดลนั้น (config ยังเลือกด้วย LOSO)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import sklearn

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    cfg = SegConfig()
    rows = load_rows(root)
    n_nobox = attach_boxes(root, rows)
    feats, cache_name = extract_features(root, rows, cfg, args.workers)

    get = lambda r, k: feats[row_key(r)][k]  # noqa: E731
    errs = [(r["path"], get(r, "err")) for r in rows if get(r, "err")]
    ok = [r for r in rows if not get(r, "err")]
    Xt = np.array([get(r, "x_train") for r in ok], np.float64)   # train view
    Xp = np.array([get(r, "x") for r in ok], np.float64)         # pipeline view
    B = np.array([get(r, "border") for r in ok], np.float64)
    y = np.array([r["label"] for r in ok])
    src = np.array([r["source"] for r in ok])
    grp = np.array([r["group"] for r in ok])
    lorig = np.array([r["label_orig"] for r in ok])
    modes = Counter((r["source"], get(r, "mode")) for r in ok)
    print(f"ใช้ได้ {len(ok)} ภาพ · error {len(errs)} · ไม่มี bbox {n_nobox}")

    report: dict = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "excluded_sources": EXCLUDED_SOURCES,
        "protocol": ("LOSO 4 fold (source) บน split=trainval label∈{light,medium,dark}; split=test ไม่ถูกโหลด · "
                     "เทรนด้วย train view (bbox สำหรับ robusta/boos, ROI agtron) · ประเมิน (ก) bbox (ข) pipeline"),
        "selection_rule": "mean LOSO macro-F1 4 fold ของ pipeline view (ปัด 2 ตำแหน่ง) → โมเดลง่ายกว่า → C เล็กกว่า",
        "seed": SEED,
        "versions": {"roastml": roastml.__version__, "python": platform.python_version(),
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "opencv": __import__("cv2").__version__, "pillow": __import__("PIL").__version__},
        "seg_config": cfg.to_dict(),
        "feature_cache": cache_name,
        "n_images": len(ok), "n_errors": len(errs), "errors": errs[:20], "n_box_rows_without_bbox": n_nobox,
        "counts_source_label": {s: dict(Counter(y[src == s])) for s in FOLDS},
        "seg_mode_by_source_pipeline": {s: {m: n for (ss, m), n in modes.items() if ss == s} for s in FOLDS},
        "feature_ms_p50_p95": [float(np.percentile([get(r, "ms") for r in ok], q)) for q in (50, 95)] if ok else [],
    }

    # ---------------- ทุก config: LOSO (เทรน train view · ประเมิน 2 view)
    def b0_fit(Xa, ya):
        return b0_spec(fit_b0(select(Xa, ["L_med"])[:, 0], ya))

    configs = [("B0", "L_med", None, 0, b0_fit)]
    for complexity, (model, fs, cs) in enumerate(B1_GRID, start=1):
        for C in cs:
            configs.append((model, fs, C, complexity, lambda Xa, ya, fs=fs, C=C: fit_b1(Xa, ya, fs, C)))

    csv_rows: list[dict] = []
    cands = []
    for model, fs, C, complexity, fit in configs:
        t0 = time.time()
        p5 = loso_predict(Xt, y, src, fit, {"bbox": Xt, "pipeline": Xp})
        g = {"model": model, "feature_set": fs, "C": C, "complexity": complexity, "pred": p5,
             "tab": {v: fold_table(y, p5[v], src) for v in p5}}
        g["score"] = g["tab"]["pipeline"]["mean"]["macro_f1"]
        cands.append(g)
        cfg_name = fs + (f"|C={C}" if C is not None else "")
        for v in ("bbox", "pipeline"):
            csv_rows += _csv_rows(model, cfg_name, v, g["tab"][v], FOLDS)
        tb, tp = g["tab"]["bbox"], g["tab"]["pipeline"]
        print(f"{model:8s} {cfg_name:14s} (ก)bbox {tb['mean']['macro_f1']:.3f} (ข)pipe {tp['mean']['macro_f1']:.3f} | "
              + " ".join(f"{s_}={tb[s_]['macro_f1']:.2f}/{tp[s_]['macro_f1']:.2f}" for s_ in FOLDS if s_ in tb)
              + f" ({time.time() - t0:.0f}s)")

    rank = lambda g: (-round(g["score"], 2), g["complexity"], g["C"] or 0)  # noqa: E731
    per_model_best = {m: sorted([g for g in cands if g["model"] == m], key=rank)[0] for m in MODELS}
    auto = sorted(cands, key=rank)[0]
    best = auto if args.export_model == "auto" else per_model_best[args.export_model]
    report["grid"] = [{"model": g["model"], "feature_set": g["feature_set"], "C": g["C"],
                       "mean_macro_f1": {v: g["tab"][v]["mean"]["macro_f1"] for v in g["tab"]},
                       "per_fold_macro_f1": {v: {s_: g["tab"][v][s_]["macro_f1"] for s_ in FOLDS if s_ in g["tab"][v]}
                                             for v in g["tab"]}}
                      for g in cands]
    for m, g in per_model_best.items():
        report[m] = {"config": {"feature_set": g["feature_set"], "C": g["C"]},
                     "folds": g["tab"],
                     "agtron_by_value": agtron_breakdown(lorig, g["pred"]["pipeline"], src)}
    report["B0"]["thresholds_per_fold"] = {s_: fit_b0(select(Xt[src != s_], ["L_med"])[:, 0], y[src != s_])
                                           for s_ in FOLDS if (src == s_).any()}
    report["auto_rule_pick"] = {"model": auto["model"], "feature_set": auto["feature_set"], "C": auto["C"],
                                "mean_macro_f1_pipeline": auto["score"]}
    report["selected"] = {"model": best["model"], "feature_set": best["feature_set"], "C": best["C"],
                          "selected_by": "auto" if args.export_model == "auto" else f"ผู้ใช้ระบุ {args.export_model}",
                          "mean_macro_f1_pipeline": best["score"],
                          "mean_macro_f1_bbox": best["tab"]["bbox"]["mean"]["macro_f1"]}

    # confusion CSV (ตัวที่ดีที่สุดของแต่ละโมเดล × view)
    cm_dir = RESULTS / "baseline_confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for name, g in per_model_best.items():
        for v, tab in g["tab"].items():
            for fold in FOLDS + ["pooled"]:
                if fold not in tab:
                    continue
                with open(cm_dir / f"{name}_{v}_{fold}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["true\\pred"] + CLASSES)
                    for c, r in zip(CLASSES, tab[fold]["cm"]):
                        w.writerow([c] + r)

    # ---------------- shortcut probe (บน pipeline view)
    probe_fs = per_model_best["B1"]["feature_set"]
    report["shortcut_probe"] = {
        s_: {"border_only": within_source_cv(B[m], y[m], grp[m]),
             f"{probe_fs}_pipeline_same_cv": within_source_cv(select(Xp[m], FEATURE_SETS[probe_fs]), y[m], grp[m]),
             "chance_macro_f1_approx": round(1 / max(len(set(y[m])), 1), 3),
             "note": {"agtron": "ใช้ ROI crop → ขอบภาพ = เมล็ด ไม่ใช่พื้นหลัง",
                      "rf_boos": "แต่ละคลาสมาจาก 2 วิดีโอ (group) → CV ภายใน source แทบไม่มีความหมาย"}.get(s_, "")}
        for s_ in FOLDS for m in [src == s_] if m.any()
    }

    # ---------------- export ตัวชนะ (เทรนบน train view ทั้ง 5 source) → ML/models/current (gitignored)
    if not args.no_export:
        if best["model"] == "B0":
            spec, backend, agree = b0_fit(Xt, y), "b0_threshold", None
        else:
            fs, C = best["feature_set"], best["C"]
            spec = fit_b1(Xt, y, fs, C)
            agree = float(np.mean(predict_spec(spec, Xt) == make_pipe(C).fit(select(Xt, FEATURE_SETS[fs]), y)
                                  .predict(select(Xt, FEATURE_SETS[fs]))))
            if agree < 0.999:
                raise RuntimeError(f"numpy/sklearn ไม่ตรงกัน: {agree}")
            backend = "b1_linear"
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        name = f"{best['model'].lower()}-{best['feature_set']}" + (f"-C{best['C']}" if best["C"] else "")
        (MODEL_OUT / "model.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        card = {
            "backend": backend, "name": name, "model_file": "model.json", "created": report["created"],
            "low_conf_threshold": LOW_CONF_THRESHOLD,
            "low_conf_threshold_note": "0 = ตอบ argmax เสมอ (ยังไม่ abstain) · confidence อยู่ใน field แยก",
            "seg_config": cfg.to_dict(), "feature_set": best["feature_set"], "C": best["C"],
            "train": {"split": "trainval", "sources": FOLDS,
                      "view": "bbox (robusta/boos) · ROI (agtron) · smart crop (ontoum224)",
                      "counts": dict(Counter(y))},
            "loso_mean_macro_f1": {"pipeline": best["score"], "bbox": best["tab"]["bbox"]["mean"]["macro_f1"]},
            "loso_folds": FOLDS, "excluded_sources": EXCLUDED_SOURCES, "selected_by": report["selected"]["selected_by"],
            "versions": report["versions"],
            "data_license_note": "เทรนรวม agtron (license Unknown) — ใช้ภายใน ห้าม commit",
        }
        (MODEL_OUT / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        report["export"] = {"dir": "ML/models/current", "name": name, "backend": backend,
                            "numpy_sklearn_agreement": agree}

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "baseline_loso.csv", csv_rows)
    (RESULTS / "baseline_loso.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
                                                encoding="utf-8")
    print(f"\nเลือก: {best['model']} {best['feature_set']} C={best['C']} · "
          f"(ข) pipeline {best['score']:.3f} · (ก) bbox {best['tab']['bbox']['mean']['macro_f1']:.3f}")
    print(f"→ {RESULTS / 'baseline_loso.csv'}\n→ {RESULTS / 'baseline_loso.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
