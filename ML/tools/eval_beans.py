"""Bean counting add-on: build candidate + measure criteria (trainval only).

Rules declared before running (user, 2026-10-08), not tuned here:
  G3 web1600 p95 <=150 ms, worst <=250 ms, <=150 blobs · G5 failure -> exact B1 result
  Phase-1 addendum: G4(b) failed in phase 0 -> no mixed_roast warning and no label override at all;
  status/label/probs/confidence must equal B1. The G2 rule (n>=4 and second share>=0.25) is only
  MEASURED here as a counterfactual false-mixed rate; it is never emitted.
Per-bean classifier = the deployed B1 model applied to each bean's core pixels (reduced scope:
rf_boos is the only source with per-bean labels, 2 videos/class).

Writes ML/models/bean_candidate (never models/current). G3 timing uses tools.bench_candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from roastml.bean_backend import VALIDATION_NOTE, BeanLinearBackend
from roastml.counter import CountConfig
from roastml.decode import decode_image
from roastml.linear_backend import LinearBackend
from tools.index_sources import read_yolo_boxes, read_yolo_names

ML_DIR = Path(__file__).resolve().parents[1]
CURRENT = ML_DIR / "models" / "current"
CANDIDATE = ML_DIR / "models" / "bean_candidate"
CLASSES = ["light", "medium", "dark"]
SOURCES = ("ontoum224", "rf_robusta", "rf_boos", "agtron", "rf_hendi")
YOLO_MAP = {"Dark Roast": "dark", "Light Roast": "light", "Medium Roast": "medium"}
G2 = {"min_beans": 4, "second_share": 0.25}  # counterfactual measurement only


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def would_trigger(prop: dict | None, n: int | None) -> bool:
    if not prop or n is None or n < G2["min_beans"]:
        return False
    return sorted(prop.values(), reverse=True)[1] >= G2["second_share"]


def check_candidate_dir(out: Path) -> None:
    """เขียนได้เฉพาะโฟลเดอร์ใหม่/ว่าง หรือ candidate b1_linear_beans เดิม (rerun ทับตัวเองได้)

    กันเขียนทับ models/current, reference_b1_r2, candidates อื่น หรือโฟลเดอร์แม่อย่าง models/
    """
    if out.resolve() == CURRENT.resolve():
        raise SystemExit("refusing to write models/current")
    if not out.exists():
        return
    if not out.is_dir():
        raise SystemExit(f"refusing to write {out}: exists and is not a directory")
    if not any(out.iterdir()):
        return
    card_path = out / "model_card.json"
    try:
        backend = json.loads(card_path.read_text(encoding="utf-8")).get("backend")
    except (OSError, ValueError, AttributeError):
        backend = None
    if backend != "b1_linear_beans":
        raise SystemExit(f"refusing to overwrite {out}: not empty and not a b1_linear_beans candidate")


def build(out: Path, run: str, extra: dict | None = None) -> dict:
    check_candidate_dir(out)
    card = json.loads((CURRENT / "model_card.json").read_text(encoding="utf-8"))
    if card.get("backend") != "b1_linear":
        raise SystemExit("models/current is not B1 b1_linear")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CURRENT / card.get("model_file", "model.json"), out / "model.json")
    new = {**card, "backend": "b1_linear_beans", "name": f"{card.get('name', 'b1')}+beans",
           "model_file": "model.json", "count_config": CountConfig().to_dict(),
           "bean_validation": VALIDATION_NOTE, "bean_run": run,
           "base_model_md5": md5(CURRENT / "model.json"), "base_card_md5": md5(CURRENT / "model_card.json"),
           "model_json_md5": md5(out / "model.json"), **(extra or {})}
    (out / "model_card.json").write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    return new


def rows_for(root: Path, sources=SOURCES, labels=CLASSES) -> list[dict]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r["split"] == "trainval" and r["source"] in sources and r["label"] in labels]


def run_one(backend, root: Path, r: dict):
    img = decode_image((root / r["path"]).read_bytes())
    if r["source"] == "agtron":  # mandatory ROI; pile mode -> no beans
        return backend.predict_with_metadata(img, source="agtron", roi=r.get("roi", ""))
    return backend.predict(img)


def vote(beans: list[dict]) -> str | None:
    if not beans:
        return None
    top = Counter(b["label"] for b in beans).most_common()
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None  # tie -> no vote


def measure(root: Path, out: Path) -> dict:
    b1 = LinearBackend(CURRENT, json.loads((CURRENT / "model_card.json").read_text(encoding="utf-8")))
    tmp = out / "candidate_measure"
    build(tmp, "measure")
    bean = BeanLinearBackend(tmp, json.loads((tmp / "model_card.json").read_text(encoding="utf-8")))
    started = time.perf_counter()
    per = []
    for r in rows_for(root):
        o1, o2 = run_one(b1, root, r), run_one(bean, root, r)
        b1_label = max(CLASSES, key=lambda c: o1.probs[c])
        per.append({"path": r["path"], "source": r["source"], "label": r["label"], "b1_label": b1_label,
                    "bean_backend_label": max(CLASSES, key=lambda c: o2.probs[c]),
                    "b1_probs_warnings_bitwise_equal": o1.probs == o2.probs and o1.warnings == o2.warnings,
                    "mixed_roast_emitted": "mixed_roast" in o2.warnings,
                    "would_trigger_mixed_G2": would_trigger(o2.proportions, o2.n_beans),
                    "n_beans": o2.n_beans, "vote_label": vote(o2.beans),
                    "n_gt": len(read_yolo_boxes(root / r["path"])) if r["source"] == "rf_boos" else None})
    summary = {}
    for s in SOURCES + ("ALL",):
        sel = [p for p in per if s == "ALL" or p["source"] == s]
        voted = [p for p in sel if p["vote_label"] is not None]
        summary[s] = {
            "n": len(sel),
            "label_flip_rate": float(np.mean([p["bean_backend_label"] != p["b1_label"] for p in sel])),
            "b1_probs_warnings_bitwise_equal": all(p["b1_probs_warnings_bitwise_equal"] for p in sel),
            "mixed_roast_emitted": int(sum(p["mixed_roast_emitted"] for p in sel)),
            "counterfactual_false_mixed_rate_G2": float(np.mean([p["would_trigger_mixed_G2"] for p in sel])),
            "n_beans_reported_rate": float(np.mean([p["n_beans"] is not None for p in sel])),
            "images_with_vote": len(voted),
            "vote_differs_from_b1_among_voted": float(np.mean([p["vote_label"] != p["b1_label"] for p in voted])) if voted else None,
            "vote_differs_from_b1_all_images": float(np.mean([p["vote_label"] is not None and p["vote_label"] != p["b1_label"] for p in sel])),
        }
    boos = [p for p in per if p["source"] == "rf_boos"]

    def count_stats(sel):
        gt = np.array([p["n_gt"] for p in sel], float)
        pr = np.array([p["n_beans"] if p["n_beans"] is not None else 0 for p in sel], float)
        err = np.abs(pr - gt)
        return {"n": len(sel), "MAE": float(err.mean()), "median_rel_err": float(np.median(err / gt)),
                "within_20pct_or_2": float(np.mean((err / gt <= .2) | (err <= 2))),
                "null_n_beans_rate": float(np.mean([p["n_beans"] is None for p in sel]))}
    count = {"rf_boos_all_modes_null_as_0": count_stats(boos),
             "rf_boos_beans_mode": count_stats([p for p in boos if p["n_beans"] is not None]),
             "note": "rf_robusta boxes are region boxes, not beans -> no count GT; no GT for dense piles"}
    names = read_yolo_names(root / "raw/rf_boos/data.yaml")
    mixed = []
    for r in rows_for(root, ("rf_boos",), ("mixed",)):
        o = run_one(bean, root, r)
        gt = [YOLO_MAP[names[b[0]]] for b in read_yolo_boxes(root / r["path"])]
        gp = np.array([gt.count(c) / len(gt) for c in CLASSES])
        rec = {"n_gt": len(gt), "n_beans": o.n_beans, "would_trigger_G2": would_trigger(o.proportions, o.n_beans),
               "mixed_roast_emitted": "mixed_roast" in o.warnings}
        if o.proportions:
            pp = np.array([o.proportions[c] for c in CLASSES])
            rec.update(L1=float(np.abs(pp - gp).sum()), dominant_correct=bool(pp.argmax() == gp.argmax()))
        mixed.append(rec)
    wp = [m for m in mixed if "L1" in m]
    mixed_summary = {"n_images": len(mixed), "groups": "1 video (rf_boos mixed)", "with_proportions": len(wp),
                     "L1_median": float(np.median([m["L1"] for m in wp])) if wp else None,
                     "L1_mean": float(np.mean([m["L1"] for m in wp])) if wp else None,
                     "dominant_acc": float(np.mean([m["dominant_correct"] for m in wp])) if wp else None,
                     "counterfactual_trigger_rate_G2": float(np.mean([m["would_trigger_G2"] for m in mixed])),
                     "mixed_roast_emitted": int(sum(m["mixed_roast_emitted"] for m in mixed)),
                     "caveat": "in-sample: deployed B1 was trained with rf_boos frames"}
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
    shutil.rmtree(tmp)
    return {"single_level": summary, "count": count, "mixed": mixed_summary, "seconds": time.perf_counter() - started}


def examples(root: Path) -> dict:
    """5 contract JSONs via roastml.api (single bean / few beans / pile / empty / broken), trainval only."""
    from roastml.api import load
    pred = load(CANDIDATE)
    rows = sorted(rows_for(root, labels=CLASSES + ["empty"]), key=lambda r: r["path"])

    def first(source, label, cond=lambda r: True):
        return next(r for r in rows if r["source"] == source and r["label"] == label and cond(r))
    picks = {"single_bean_ontoum224": first("ontoum224", "medium"), "few_beans_rf_boos": first("rf_boos", "dark"),
             "pile_rf_robusta": first("rf_robusta", "dark", lambda r: ":video:" not in r["group"]),
             "empty_rf_hendi": first("rf_hendi", "empty")}
    out = {k: {"path": r["path"], "result": pred.predict_bytes((root / r["path"]).read_bytes())} for k, r in picks.items()}
    out["broken_bytes"] = {"path": None, "result": pred.predict_bytes(os.urandom(2048))}
    return out


def loso_vote(root: Path) -> dict:
    """Image-level majority vote of per-bean labels under LOSO (report only; never changes labels)."""
    from roastml.features import FEATURE_SETS
    from roastml.linear_model import LinearSoftmax
    from tools.compare_tabular import load_cached
    from tools.train_baseline import FOLDS, fold_table, make_pipe, spec_from_sklearn
    reference = json.loads((ML_DIR / "results/baseline_loso.json").read_text(encoding="utf-8"))
    rows, Xt, _, cap, _, _ = load_cached(root, reference)  # Xt is already the 15 Lab_hist columns
    names = FEATURE_SETS["Lab_hist"]
    if Xt.shape[1] != len(names):
        raise ValueError("cached training features are not the Lab_hist set")
    src, y = np.array([r["source"] for r in rows]), np.array([r["label"] for r in rows])
    tmp = ML_DIR / "models" / "_bean_loso_tmp"
    build(tmp, "loso_vote")
    bean = BeanLinearBackend(tmp, json.loads((tmp / "model_card.json").read_text(encoding="utf-8")))
    b1_pred, vote_pred = np.empty(len(rows), object), np.empty(len(rows), object)
    n_voted = 0
    for fold in FOLDS:
        tr = cap & (src != fold)
        pipe = make_pipe(0.01).fit(Xt[tr], y[tr])
        bean.model = LinearSoftmax(spec_from_sklearn(pipe, names, CLASSES))
        for i in np.flatnonzero(src == fold):
            o = run_one(bean, root, rows[i])
            b1_pred[i] = max(CLASSES, key=lambda c: o.probs[c])
            v = vote(o.beans)
            n_voted += v is not None
            vote_pred[i] = v if v is not None else b1_pred[i]
    shutil.rmtree(tmp)
    t_b1, t_v = fold_table(y, b1_pred.astype(str), src), fold_table(y, vote_pred.astype(str), src)
    return {"b1_mean_macro_f1": t_b1["mean"]["macro_f1"], "vote_mean_macro_f1": t_v["mean"]["macro_f1"],
            "b1_pooled_macro_f1": t_b1["pooled"]["macro_f1"], "vote_pooled_macro_f1": t_v["pooled"]["macro_f1"],
            "per_fold": {f: {"b1": t_b1[f]["macro_f1"], "vote": t_v[f]["macro_f1"]} for f in FOLDS},
            "images_with_vote": n_voted, "n": len(rows),
            "note": "B1 fold models refit (C=0.01, R2 cap) from cached features; no vote -> B1 label"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ML_DIR / "results" / "beans_phase1_20261008")
    args = ap.parse_args(argv)
    root = args.data_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    before = {"model.json": md5(CURRENT / "model.json"), "model_card.json": md5(CURRENT / "model_card.json")}
    rep = measure(root, args.out)
    (args.out / "measure.partial.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    rep["loso_vote"] = loso_vote(root)
    final = build(CANDIDATE, "beans_phase1_20261008", {"measured": {
        "b1_identical_all_single_level_trainval": rep["single_level"]["ALL"]["b1_probs_warnings_bitwise_equal"],
        "counterfactual_false_mixed_rate_G2": rep["single_level"]["ALL"]["counterfactual_false_mixed_rate_G2"],
        "count_rf_boos_beans_mode": rep["count"]["rf_boos_beans_mode"],
        "report": "ML/results/beans_phase1_20261008/report.json"}})
    rep["candidate"] = {"dir": str(CANDIDATE), "model_json_md5": final["model_json_md5"]}
    rep["examples"] = examples(root)
    rep["current_md5_before_after"] = [before, {"model.json": md5(CURRENT / "model.json"),
                                                "model_card.json": md5(CURRENT / "model_card.json")}]
    rep["versions"] = {"python": platform.python_version(), "numpy": np.__version__}
    rep["frozen_test_loaded"] = False
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("count", "mixed", "loso_vote", "candidate")}, indent=1, ensure_ascii=False))
    print(json.dumps(rep["single_level"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
