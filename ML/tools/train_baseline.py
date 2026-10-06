"""Baseline B0/B1 + LOSO + shortcut probe → ML/results/baseline_loso.{csv,json} และ export B1 → ML/models/b1/

โปรโตคอล (ml-spec ข้อ 5, 7, 8 + results/split_decisions.md):
- ข้อมูล: manifest แถว split=trainval และ label ∈ {light, medium, dark} เท่านั้น · **ไม่โหลด split=test เลย**
- LOSO 6 fold หน่วย = source: ontoum224, rf_hendi, rf_devlong, rf_robusta, rf_boos, agtron (5 เครื่อง)
- รูป: อ่าน bytes → roastml.decode.decode_image (เส้นทางเดียวกับ Pi) → agtron crop ROI (ไม่ smart crop) · อื่นๆ segment/smart crop
  (segment ไม่เจอเมล็ด → ใช้ภาพเต็ม)
- B0: threshold 2 ค่าบน L_med (เลือกบน train fold ด้วย macro-F1) · B1: StandardScaler + LogisticRegression(class_weight=balanced)
  เลือก feature set × C ด้วย mean LOSO macro-F1 (ตัวเลข fold ของ config ที่ถูกเลือกจึงมองโลกในแง่ดีเล็กน้อย)
- shortcut probe: ทายคลาสจาก "ขอบภาพ" อย่างเดียว ภายในแต่ละ source (StratifiedGroupKFold ตาม group)

ใช้: python -m tools.train_baseline [--workers 8] [--no-export]
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
from roastml.features import BORDER_NAMES, FEATURE_SETS, FEATURES_ALL, border_features, image_features, select
from roastml.linear_model import LinearSoftmax, Threshold3, spec_from_sklearn
from roastml.paths import CACHE, data_dir
from roastml.segment import SegConfig

ML_DIR = Path(__file__).resolve().parent.parent
RESULTS = ML_DIR / "results"
MODEL_OUT = ML_DIR / "models" / "b1"
CLASSES = ["light", "medium", "dark"]          # ลำดับใน report
ORDINAL = {"dark": 0, "medium": 1, "light": 2}  # สำหรับ B0 (L* ต่ำ = เข้ม)
FOLDS = ["ontoum224", "rf_hendi", "rf_devlong", "rf_robusta", "rf_boos", "agtron"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
B1_SETS = ["L", "Lab", "Lab_hist"]
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
def _extract_one(job: tuple[str, str, dict]) -> dict:
    path, roi, cfg = job
    try:
        img = decode_image(Path(path).read_bytes())
        rgb = np.asarray(img.image)
        if roi:  # agtron: พิกัด ROI อ้างอิงภาพเต็มหลังหมุน EXIF → สเกลตามภาพที่ decode แล้ว
            s = img.scale
            x1, y1, x2, y2 = (int(round(int(v) * s)) for v in roi.split())
            rgb = rgb[y1:y2, x1:x2]
        f = image_features(rgb, SegConfig.from_dict(cfg), find_beans=not roi)
        return {"x": f.x, "border": border_features(rgb), "mode": f.seg.mode, "wb": f.seg.wb_applied,
                "ms": f.ms, "err": ""}
    except Exception as e:  # รายงานรวมทีหลัง ไม่ให้ทั้งรอบล้ม
        return {"err": f"{type(e).__name__}: {e}"}


def extract_features(root: Path, rows: list[dict], cfg: SegConfig, workers: int) -> dict:
    cfg_d = cfg.to_dict()
    tag = hashlib.md5(json.dumps({"cfg": cfg_d, "v": roastml.__version__, "feat": FEATURES_ALL,
                                  "border": BORDER_NAMES}, sort_keys=True).encode()).hexdigest()[:10]
    cache_path = root / CACHE / f"baseline_features_{tag}.json"
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"cache เสีย ({e}) → สร้างใหม่")
    key = lambda r: f"{r['md5']}|{r['roi']}"  # noqa: E731
    todo = [r for r in rows if key(r) not in cache]
    if todo:
        print(f"สกัด feature {len(todo)} ภาพ (cache {len(rows) - len(todo)}) workers={workers}")
        t0 = time.time()
        jobs = [(str(root / r["path"]), r["roi"], cfg_d) for r in todo]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_extract_one, jobs, chunksize=16))
        else:
            res = [_extract_one(j) for j in jobs]
        for r, e in zip(todo, res):
            if not e["err"]:
                e = {**e, "x": e["x"].tolist(), "border": e["border"].tolist()}
            cache[key(r)] = e
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, cache_path)
        print(f"  เสร็จใน {time.time() - t0:.0f}s → {cache_path.name}")
    return {key(r): cache[key(r)] for r in rows}, cache_path.name


# ================================================================ LOSO
def loso_predict(X: np.ndarray, y: np.ndarray, src: np.ndarray, fit) -> np.ndarray:
    """fit(X_tr, y_tr) → spec · คืน prediction ของทุกแถวตอนที่ source นั้นถูก hold out"""
    pred = np.empty(len(y), dtype=object)
    for s in FOLDS:
        te = src == s
        if not te.any():
            continue
        spec = fit(X[~te], y[~te])
        pred[te] = predict_spec(spec, X[te])
    return pred.astype(str)


def fold_table(y, p, src) -> dict:
    out = {s: metrics(y[src == s], p[src == s]) for s in FOLDS if (src == s).any()}
    keys = ["acc", "macro_f1", "cross_step_rate"]
    out["mean"] = {k: float(np.mean([out[s][k] for s in FOLDS if s in out])) for k in keys}
    out["pooled"] = metrics(y, p)
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
    return [r for r in rows if r["split"] == "trainval" and r["label"] in CLASSES]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import sklearn

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    cfg = SegConfig()
    rows = load_rows(root)
    feats, cache_name = extract_features(root, rows, cfg, args.workers)

    errs = [(r["path"], feats[f"{r['md5']}|{r['roi']}"]["err"]) for r in rows if feats[f"{r['md5']}|{r['roi']}"]["err"]]
    ok = [r for r in rows if not feats[f"{r['md5']}|{r['roi']}"]["err"]]
    get = lambda r, k: feats[f"{r['md5']}|{r['roi']}"][k]  # noqa: E731
    X = np.array([get(r, "x") for r in ok], np.float64)
    B = np.array([get(r, "border") for r in ok], np.float64)
    y = np.array([r["label"] for r in ok])
    src = np.array([r["source"] for r in ok])
    grp = np.array([r["group"] for r in ok])
    lorig = np.array([r["label_orig"] for r in ok])
    modes = Counter((r["source"], get(r, "mode")) for r in ok)
    print(f"ใช้ได้ {len(ok)} ภาพ · error {len(errs)}")

    report: dict = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "LOSO 6 fold (source) บน split=trainval label∈{light,medium,dark}; split=test ไม่ถูกโหลด",
        "seed": SEED,
        "versions": {"roastml": roastml.__version__, "python": platform.python_version(),
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "opencv": __import__("cv2").__version__, "pillow": __import__("PIL").__version__},
        "seg_config": cfg.to_dict(),
        "feature_cache": cache_name,
        "n_images": len(ok),
        "errors": errs[:20],
        "n_errors": len(errs),
        "counts_source_label": {s: dict(Counter(y[src == s])) for s in FOLDS},
        "seg_mode_by_source": {s: {m: n for (ss, m), n in modes.items() if ss == s} for s in FOLDS},
        "wb_applied_by_source": {s: int(sum(get(r, "wb") for r in ok if r["source"] == s)) for s in FOLDS},
        "feature_ms_p50_p95": [float(np.percentile([get(r, "ms") for r in ok], q)) for q in (50, 95)],
    }

    csv_rows: list[dict] = []

    def add_csv(model: str, config: str, tab: dict) -> None:
        for fold in FOLDS + ["mean", "pooled"]:
            if fold not in tab:
                continue
            t = tab[fold]
            row = {"model": model, "config": config, "fold": fold, "n": t.get("n", ""),
                   "acc": round(t["acc"], 4), "macro_f1": round(t["macro_f1"], 4),
                   "cross_step_rate": round(t["cross_step_rate"], 4)}
            for c in CLASSES:
                pc = t.get("per_class", {}).get(c, {})
                for k in ("precision", "recall", "f1"):
                    row[f"{c}_{k[0] if k != 'f1' else 'f1'}"] = round(pc[k], 4) if pc else ""
            csv_rows.append(row)

    # ---------------- B0
    t0 = time.time()
    p_b0 = loso_predict(X, y, src, lambda Xt, yt: b0_spec(fit_b0(select(Xt, ["L_med"])[:, 0], yt)))
    b0 = fold_table(y, p_b0, src)
    b0_thr = {s: fit_b0(select(X[src != s], ["L_med"])[:, 0], y[src != s]) for s in FOLDS}
    report["B0"] = {"config": "L_med thresholds", "folds": b0, "thresholds_per_fold": b0_thr,
                    "agtron_by_value": agtron_breakdown(lorig, p_b0, src)}
    add_csv("B0", "L_med", b0)
    print(f"B0 mean LOSO macro-F1 {b0['mean']['macro_f1']:.3f} ({time.time() - t0:.0f}s)")

    # ---------------- B1 grid
    grid = []
    preds = {}
    for fs in B1_SETS:
        for C in C_GRID:
            p = loso_predict(X, y, src, lambda Xt, yt, fs=fs, C=C: fit_b1(Xt, yt, fs, C))
            tab = fold_table(y, p, src)
            preds[(fs, C)] = p
            grid.append({"feature_set": fs, "C": C, "n_features": len(FEATURE_SETS[fs]),
                         "mean_macro_f1": tab["mean"]["macro_f1"], "mean_acc": tab["mean"]["acc"],
                         "per_fold_macro_f1": {s: tab[s]["macro_f1"] for s in FOLDS if s in tab}, "tab": tab})
            add_csv("B1", f"{fs}|C={C}", tab)
            print(f"B1 {fs:8s} C={C:<5} mean macro-F1 {tab['mean']['macro_f1']:.3f} "
                  + " ".join(f"{s}={tab[s]['macro_f1']:.2f}" for s in FOLDS if s in tab))
    # เลือก: macro-F1 เฉลี่ยสูงสุด (ปัด 2 ตำแหน่ง) → feature น้อยกว่า → C เล็กกว่า (regularize แรงกว่า)
    best = sorted(grid, key=lambda g: (-round(g["mean_macro_f1"], 2), g["n_features"], g["C"]))[0]
    bfs, bC = best["feature_set"], best["C"]
    p_b1 = preds[(bfs, bC)]
    report["B1"] = {
        "selection": "mean LOSO macro-F1 (ปัด 2 ตำแหน่ง) → feature น้อยกว่า → C เล็กกว่า",
        "selected": {"feature_set": bfs, "C": bC},
        "grid": [{k: v for k, v in g.items() if k != "tab"} for g in grid],
        "folds": best["tab"],
        "agtron_by_value": agtron_breakdown(lorig, p_b1, src),
    }

    # confusion CSV
    cm_dir = RESULTS / "baseline_confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for name, tab in (("B0", b0), ("B1", best["tab"])):
        for fold in FOLDS + ["pooled"]:
            if fold in tab:
                with open(cm_dir / f"{name}_{fold}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["true\\pred"] + CLASSES)
                    for c, r in zip(CLASSES, tab[fold]["cm"]):
                        w.writerow([c] + r)

    # ---------------- shortcut probe
    probe = {}
    for s in FOLDS:
        m = src == s
        probe[s] = {
            "border_only": within_source_cv(B[m], y[m], grp[m]),
            "b1_features_same_cv": within_source_cv(select(X[m], FEATURE_SETS[bfs]), y[m], grp[m]),
            "n_classes": len(set(y[m])),
            "chance_macro_f1_approx": round(1 / max(len(set(y[m])), 1), 3),
            "note": "agtron ใช้ ROI crop → ขอบภาพ = เมล็ด ไม่ใช่พื้นหลัง" if s == "agtron" else "",
        }
    report["shortcut_probe"] = probe

    # ---------------- export B1 (เทรนบน trainval ทั้งหมดของ 6 source)
    if not args.no_export:
        spec = fit_b1(X, y, bfs, bC)
        agree = float(np.mean(predict_spec(spec, X) == make_pipe(bC).fit(select(X, FEATURE_SETS[bfs]), y)
                              .predict(select(X, FEATURE_SETS[bfs]))))
        if agree < 0.999:
            raise RuntimeError(f"numpy/sklearn ไม่ตรงกัน: {agree}")
        MODEL_OUT.mkdir(parents=True, exist_ok=True)
        name = f"b1-{bfs}-C{bC}"
        (MODEL_OUT / "model.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        card = {
            "backend": "b1_linear", "name": name, "model_file": "model.json",
            "created": report["created"], "low_conf_threshold": 0.6,
            "low_conf_threshold_note": "ค่าเริ่มต้น ยังไม่ได้เลือกด้วย LOSO",
            "seg_config": cfg.to_dict(), "feature_set": bfs, "C": bC,
            "train": {"split": "trainval", "sources": FOLDS, "counts": dict(Counter(y))},
            "loso_mean_macro_f1": best["mean_macro_f1"], "versions": report["versions"],
            "data_license_note": "เทรนรวม agtron (license Unknown) — ใช้ภายใน",
        }
        (MODEL_OUT / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        report["export"] = {"dir": "ML/models/b1", "name": name, "numpy_sklearn_agreement": agree}

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "baseline_loso.csv", csv_rows)
    (RESULTS / "baseline_loso.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
                                                encoding="utf-8")
    print(f"\nเลือก B1: {bfs} C={bC} · mean LOSO macro-F1 {best['mean_macro_f1']:.3f}")
    print(f"→ {RESULTS / 'baseline_loso.csv'}\n→ {RESULTS / 'baseline_loso.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
