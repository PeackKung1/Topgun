"""Shared train/serve bean features and deterministic brightness grouping.

Runtime depends only on numpy/OpenCV. Group thresholds are selected using
synthetic mixtures of training-fold beans, never the held-out source.
"""
from __future__ import annotations

import itertools
import numpy as np

from .counter import CountConfig, count_beans
from .features import FEATURES_ALL, pixel_stats_grouped
from .segment import SegConfig, PreparedImage


def count_features(result):
    if not result.n:
        return np.zeros((0, len(FEATURES_ALL)), np.float64)
    return pixel_stats_grouped(result.lab[result.core], result.labels[result.core]-1, result.n)


def bean_features(rgb, count_config: CountConfig, seg_config: SegConfig, *, force_pile=False,
                  prepared: PreparedImage | None = None):
    result = count_beans(rgb, count_config, seg_config, force_pile=force_pile, prepared=prepared)
    return result, count_features(result)


def brightness_groups(X, k):
    """1D k-means with deterministic quantile starts; no process-global RNG."""
    v = np.asarray(X)[:, FEATURES_ALL.index("L_med")]
    centers = np.percentile(v, (np.arange(k)+0.5)*100/k)
    for _ in range(30):
        labels = np.abs(v[:, None]-centers).argmin(1)
        if len(np.unique(labels)) < k:
            return None, None
        new = np.array([v[labels == j].mean() for j in range(k)])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


def group_predictions(X, P, classes, config=None):
    """k=1..3; labels use group mean probabilities with strict dark→light order."""
    config = config or {}
    if set(config)-{"min_L_gap", "max_k"}:
        raise ValueError("unknown group_config key")
    gap = float(config.get("min_L_gap", 12.0))
    max_k = config.get("max_k", 3)
    if not np.isfinite(gap) or gap <= 0 or max_k not in (1, 2, 3):
        raise ValueError("invalid group_config")
    if not len(X):
        return np.array([], int), np.array([], float), {"k": 0}
    selected, k = np.zeros(len(X), int), 1
    for candidate in range(2, min(max_k, len(X))+1):
        ll, centers = brightness_groups(X, candidate)
        if ll is not None and np.min(np.diff(centers)) >= gap:
            selected, k = ll, candidate
    mean = np.array([P[selected == j].mean(0) for j in range(k)])
    ordered = [classes.index(c) for c in ("dark", "medium", "light")]
    # Enumerate ordered distinct class assignments (3, 3, 1 possibilities).
    choices = list(itertools.combinations(ordered, k))
    mapping = max(choices, key=lambda cols: sum(np.log(max(mean[j, c], 1e-12)) for j, c in enumerate(cols)))
    labels = np.asarray(mapping)[selected]
    # Preserve the specified .999 proportion + .001 mean-prob formula even
    # in the 1000-bean, one-vote-margin corner case. An ordered forced mapping
    # can otherwise let the mean probability overturn the count majority.
    counts = np.bincount(labels, minlength=len(classes))
    aggregate = .999*counts/len(labels) + .001*P.mean(0)
    if counts[aggregate.argmax()] < counts.max():
        best = int(P.mean(0).argmax())
        return np.full(len(X), best, int), np.full(len(X), P.mean(0)[best]), {"k": 1}
    confidence = np.array([mean[j, mapping[j]] for j in selected])
    return labels, confidence, {"k": k}


def majority_probs(labels, P, classes):
    n = len(labels)
    if not n:
        raise ValueError("empty beans need B1 fallback")
    counts = np.bincount(labels, minlength=len(classes))
    proportions = counts / n
    probs = 0.999 * proportions + 0.001 * P.mean(0)
    if counts[probs.argmax()] < counts.max():
        raise ValueError("mean probability overturns bean majority; unsupported count/mapping")
    return ({c: float(probs[i]) for i,c in enumerate(classes)},
            {c: float(proportions[i]) for i,c in enumerate(classes)})
