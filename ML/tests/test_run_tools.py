"""Synthetic tests for the 2026-10-08 follow-up tools (no dataset, no torch)."""
import csv

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="training tools need scikit-learn; not installed on the Pi runtime")

from tools import summarize_runs
from tools.d1_frozen_probe import PLAN as PROBE_PLAN, select
from tools.train_d1 import AUG_MEDIUM, PLAN, PLAN_MEDAUG, PROTOCOLS, augment_one, augmentation_params


def test_default_augmentation_is_the_predeclared_mild_one():
    image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    a = augment_one(image, np.random.default_rng(7))
    b = augment_one(image, np.random.default_rng(7), PLAN["augmentation_mild"])
    assert np.array_equal(a, b)


def test_medium_augmentation_is_valid_and_predeclared_only():
    image = np.full((64, 64, 3), 120, np.uint8)
    out = augment_one(image, np.random.default_rng(1), AUG_MEDIUM)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert augmentation_params(PLAN_MEDAUG, "none") is None
    assert augmentation_params(PLAN_MEDAUG, "medium") == AUG_MEDIUM
    with pytest.raises(ValueError):
        augmentation_params(PLAN, "medium")  # groupguard never declared medium


def test_medaug_protocol_changes_only_finetune_fields():
    changed = {k for k in set(PLAN) | set(PLAN_MEDAUG) if PLAN.get(k) != PLAN_MEDAUG.get(k)}
    assert changed == {"protocol", "finetune_grid", "finetune_max_epochs", "augmentation_medium",
                       "families", "comparison"}
    assert PROTOCOLS["groupguard"]["plan"] is PLAN
    for forbidden in ("hue", "saturation", "gray", "equal"):
        assert not any(forbidden in k for k in AUG_MEDIUM)


def test_frozen_probe_selection_is_group_disjoint_and_deterministic():
    rng = np.random.default_rng(0)
    y = np.repeat(["light", "medium", "dark"], 40)
    X = rng.normal(size=(120, 5)) + np.repeat(np.eye(3, 5) * 3, 40, axis=0)
    groups = np.array([f"g{i // 2}" for i in range(120)])
    best1, info1 = select(X, y, groups, 1)
    best2, _ = select(X, y, groups, 1)
    assert best1 == best2 and best1["C"] in PROBE_PLAN["C_grid"]
    assert info1["trials"][info1["selected"]]["validation_macro_f1"] > 0.8


def _write(path, rows, prefix):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", *[prefix + c for c in ("light", "medium", "dark")]])
        w.writerows(rows)


def test_summarize_refuses_different_holdout_populations(tmp_path, monkeypatch):
    sources = ["ontoum224", "rf_robusta", "rf_boos", "agtron"]
    base = [[f"img{i}", sources[i % 4], ["light", "medium", "dark"][i % 3], .6, .3, .1] for i in range(24)]
    _write(tmp_path / "a.csv", base, "prob_")
    _write(tmp_path / "b.csv", base[:-1], "prob_")
    monkeypatch.setattr(summarize_runs, "R", tmp_path)
    monkeypatch.setattr(summarize_runs, "MODELS", {"A": (["a.csv"], "prob_", "a"), "B": (["b.csv"], "prob_", "b")})
    with pytest.raises(ValueError, match="holdout population differs"):
        summarize_runs.main(["--out", str(tmp_path / "out")])


def test_summarize_metrics_match_hand_count(tmp_path, monkeypatch):
    sources = ["ontoum224", "rf_robusta", "rf_boos", "agtron"]
    rows = [[f"img{i}", sources[i % 4], "light", .7, .2, .1] for i in range(8)]
    _write(tmp_path / "a.csv", rows, "prob_")
    monkeypatch.setattr(summarize_runs, "R", tmp_path)
    monkeypatch.setattr(summarize_runs, "MODELS", {"A": (["a.csv"], "prob_", "a")})
    monkeypatch.setattr(summarize_runs, "plot", lambda *a, **k: None)
    summarize_runs.main(["--out", str(tmp_path / "out")])
    with (tmp_path / "out" / "model_comparison.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert float(row["pooled_acc_mean"]) == 1.0 and float(row["pooled_cross_step_rate_mean"]) == 0.0
