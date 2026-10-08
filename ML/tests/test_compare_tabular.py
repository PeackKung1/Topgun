from __future__ import annotations

import numpy as np
import pytest

from roastml.features import FEATURES_ALL, FEATURE_SETS
from roastml.tabular_backend import NumpyTabularModel
from tools.compare_tabular import (
    CLASSES, calibrate_prob, calibration, estimator_probs, export_spec,
    fit_estimator, inner_split, load_inner_group_guard, predict_export, qa_keep,
)


def data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 15))
    y = np.array(CLASSES * 30)
    # Every independent group has all classes; images within group stay together.
    groups = np.repeat([f"g{i}" for i in range(30)], 3)
    X[:, 0] += np.tile([2.0, 0.0, -2.0], 30)
    return X, y, groups


def full(X):
    a = np.zeros((len(X), len(FEATURES_ALL)))
    a[:, [FEATURES_ALL.index(n) for n in FEATURE_SETS["Lab_hist"]]] = X
    return a


def test_inner_split_disjoint_and_reproducible():
    X, y, groups = data()
    tr, va, info = inner_split(y, groups, 12)
    assert not (set(groups[tr]) & set(groups[va]))
    assert set(y[tr]) == set(y[va]) == set(CLASSES)
    tr2, va2, _ = inner_split(y, groups, 12)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(va, va2)


def test_group_missing_class_never_falls_back_to_images():
    y = np.array(["light"] * 5 + ["medium"] * 5 + ["dark"] * 5)
    groups = y.copy()
    with pytest.raises(ValueError, match="every group fold misses a class"):
        inner_split(y, groups, 12)


@pytest.mark.parametrize("family,cfg", [
    ("B1", {"C": 0.01}),
    ("MLP", {"hidden": 16, "alpha": 0.1, "epochs": 10}),
    ("RandomForest", {"n_estimators": 4, "max_depth": 4, "min_samples_leaf": 1}),
    ("SVM_RBF", {"C": 1.0, "gamma": "scale"}),
])
def test_numpy_export_probability_and_class_order_parity(family, cfg):
    X, y, _ = data()
    pipe, _ = fit_estimator(family, cfg, 5, X, y)
    spec = export_spec(pipe, family, 1.5)
    native = estimator_probs(pipe, family, X, 1.5)
    actual = predict_export(spec, full(X))
    np.testing.assert_allclose(actual, native, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(actual.argmax(axis=1), native.argmax(axis=1))
    np.testing.assert_allclose(NumpyTabularModel(spec).proba(full(X)), native, atol=1e-12, rtol=1e-12)
    assert spec["classes"] == CLASSES


def test_calibration_keeps_labels_and_normalization():
    p = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
    calibrated = calibrate_prob(p, 2.0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    np.testing.assert_array_equal(calibrated.argmax(axis=1), p.argmax(axis=1))
    report = calibration(np.array(["dark", "medium"]), calibrated)
    assert sum(r["n"] for r in report["reliability"]) == 2
    assert 0 <= report["ece"] <= 1


def test_saturated_mlp_probability_calibration_matches_native():
    X, y, _ = data()
    pipe, _ = fit_estimator("MLP", {"hidden": 16, "alpha": 0.1, "epochs": 10}, 5, X, y)
    extreme = X * 10000
    spec = export_spec(pipe, "MLP", 5.0)
    native = estimator_probs(pipe, "MLP", extreme, 5.0)
    np.testing.assert_allclose(predict_export(spec, full(extreme)), native, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(NumpyTabularModel(spec).proba(full(extreme)), native, atol=1e-12, rtol=1e-12)


def test_qa_requires_complete_coverage_and_only_hard_exclusion():
    rows = [{"path": "a"}, {"path": "b"}]
    np.testing.assert_array_equal(qa_keep(rows, {"a": {"hard_exclude": False, "flags": ["blurry"]},
                                              "b": {"hard_exclude": True}}), [True, False])
    with pytest.raises(ValueError, match="cover every"):
        qa_keep(rows, {"a": False})


def test_group_guard_rejects_residual_near_duplicates(tmp_path):
    import json
    rows = [{"path": "a", "group": "g1"}, {"path": "b", "group": "g2"}]
    (tmp_path / "near_duplicate_pairs.csv").write_text("a,b\na,b\n", encoding="utf-8")
    guard = tmp_path / "inner_groups.json"
    guard.write_text(json.dumps({"group_to_inner_group": {"g1": "g1", "g2": "g2"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="residual near-duplicate"):
        load_inner_group_guard(rows, guard)
    guard.write_text(json.dumps({"group_to_inner_group": {"g1": "connected", "g2": "connected"}}), encoding="utf-8")
    groups, evidence = load_inner_group_guard(rows, guard)
    np.testing.assert_array_equal(groups, ["connected", "connected"])
    assert evidence["reference_population_pairs_guarded"] == 1
    assert evidence["residual_pairs_crossing_any_inner_split"] == 0
