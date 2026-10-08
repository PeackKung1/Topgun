"""Portable feature classifiers; deployment never imports torch or sklearn."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import LABELS
from .features import FEATURES_ALL
from .linear_backend import LinearBackend
from .linear_model import LinearSoftmax
from .segment import SegConfig


def softmax(logits):
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def model_file_path(model_dir: Path, filename: str) -> Path:
    root = Path(model_dir).resolve()
    p = (root / filename).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError("model_file must be an existing file inside the model directory")
    return p


class NumpyTabularModel:
    def __init__(self, spec: dict):
        self.spec = spec
        self.kind = spec["type"]
        self.classes = list(spec["classes"])
        if len(self.classes) != 3 or set(self.classes) != set(LABELS):
            raise ValueError("classes must contain each contract class exactly once")
        self.feature_names = list(spec["feature_names"])
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique and nonempty")
        self.idx = np.array([FEATURES_ALL.index(n) for n in self.feature_names])
        d = len(self.idx)
        self.mean = np.asarray(spec.get("scaler_mean", [0.] * d), np.float64)
        self.scale = np.asarray(spec.get("scaler_scale", [1.] * d), np.float64)
        self.temperature = float(spec.get("temperature", 1.0))
        if self.mean.shape != (d,) or self.scale.shape != (d,) or not np.all(np.isfinite(self.mean)) \
                or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0) \
                or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid scaler or temperature")
        if self.kind == "softmax":
            self.linear = LinearSoftmax(spec)
        elif self.kind == "mlp":
            if spec.get("activation", "relu") != "relu":
                raise ValueError("only ReLU MLP export is supported")
            self.layers = []
            width = d
            for layer in spec["layers"]:
                W, b = np.asarray(layer["W"], float), np.asarray(layer["b"], float)
                if W.ndim != 2 or W.shape[0] != width or b.shape != (W.shape[1],) \
                        or not np.isfinite(W).all() or not np.isfinite(b).all():
                    raise ValueError("invalid MLP layer")
                self.layers.append((W, b))
                width = W.shape[1]
            if not self.layers or width != 3:
                raise ValueError("MLP must output three logits")
        elif self.kind == "forest":
            self.trees = []
            for tree in spec["trees"]:
                left, right = np.asarray(tree["children_left"], int), np.asarray(tree["children_right"], int)
                feat, threshold = np.asarray(tree["feature"], int), np.asarray(tree["threshold"], float)
                values = np.asarray(tree["value"], float)
                if values.ndim == 3 and values.shape[1] == 1:
                    values = values[:, 0, :]
                n = len(left)
                if not n or right.shape != (n,) or feat.shape != (n,) or threshold.shape != (n,) \
                        or values.shape != (n, 3) or not np.isfinite(values).all() \
                        or np.any(values < 0) or np.any(values.sum(1) <= 0):
                    raise ValueError("invalid forest tree arrays")
                for node in range(n):
                    if left[node] == right[node] == -1:
                        continue
                    if not (node < left[node] < n and node < right[node] < n and 0 <= feat[node] < d):
                        raise ValueError("invalid tree edge or feature")
                self.trees.append((left, right, feat, threshold, values / values.sum(1, keepdims=True)))
            if not self.trees:
                raise ValueError("forest must contain trees")
        elif self.kind == "svm_rbf":
            self.estimator_classes = list(spec["estimator_classes"])
            if len(self.estimator_classes) != 3 or set(self.estimator_classes) != set(LABELS):
                raise ValueError("invalid estimator class order")
            self.support = np.asarray(spec["support_vectors"], float)
            self.n_support = np.asarray(spec["n_support"], int)
            self.dual = np.asarray(spec["dual_coef"], float)
            self.intercept = np.asarray(spec["intercept"], float)
            self.gamma = float(spec["gamma"])
            n = len(self.support)
            if self.support.shape != (n, d) or self.n_support.shape != (3,) \
                    or np.any(self.n_support < 0) or self.n_support.sum() != n \
                    or self.dual.shape != (2, n) or self.intercept.shape != (3,) \
                    or not np.isfinite(self.support).all() or not np.isfinite(self.dual).all() \
                    or not np.isfinite(self.intercept).all() or not np.isfinite(self.gamma) or self.gamma <= 0:
                raise ValueError("invalid RBF SVM arrays")
        else:
            raise ValueError(f"unsupported tabular model: {self.kind}")

    def proba(self, X_all):
        if self.kind == "softmax":
            p = self.linear.proba(X_all)
            return softmax(np.log(np.maximum(p, 1e-15)) / self.temperature)
        X = np.atleast_2d(np.asarray(X_all, float))[:, self.idx]
        X = (X - self.mean) / self.scale
        if not np.isfinite(X).all():
            raise ValueError("nonfinite input features")
        if self.kind == "mlp":
            for i, (W, b) in enumerate(self.layers):
                X = X @ W + b
                if i + 1 < len(self.layers):
                    X = np.maximum(X, 0)
            # Match exported sklearn probability calibration, including clipping
            # at saturated logits, rather than recalibrating unclipped logits.
            raw = softmax(X)
            return softmax(np.log(np.maximum(raw, 1e-15)) / self.temperature)
        if self.kind == "forest":
            X = X.astype(np.float32)  # sklearn tree traversal compares float32 input.
            P = np.zeros((len(X), 3), float)
            for left, right, feat, threshold, values in self.trees:
                nodes = np.zeros(len(X), int)
                for _ in range(len(left)):
                    active = np.flatnonzero(left[nodes] != -1)
                    if not len(active):
                        break
                    node = nodes[active]
                    go_left = X[active, feat[node]] <= threshold[node]
                    nodes[active] = np.where(go_left, left[node], right[node])
                P += values[nodes]
            P /= len(self.trees)
            return softmax(np.log(np.maximum(P, 1e-15)) / self.temperature)
        distance = np.maximum((X ** 2).sum(1)[:, None] + (self.support ** 2).sum(1)[None, :]
                              - 2 * X @ self.support.T, 0)
        kernel = np.exp(-self.gamma * distance)
        offsets = np.r_[0, np.cumsum(self.n_support)]
        votes, confidence = np.zeros((len(X), 3)), np.zeros((len(X), 3))
        pair = 0
        for i in range(3):
            for j in range(i + 1, 3):
                si, sj = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
                dec = kernel[:, si] @ self.dual[j - 1, si] + kernel[:, sj] @ self.dual[i, sj] + self.intercept[pair]
                votes[:, i] += dec >= 0
                votes[:, j] += dec < 0
                confidence[:, i] += dec
                confidence[:, j] -= dec
                pair += 1
        logits = votes + confidence / (3 * (np.abs(confidence) + 1))
        order = [self.estimator_classes.index(c) for c in self.classes]
        return softmax(logits[:, order] / self.temperature)


class TabularBackend(LinearBackend):
    """Same feature/ROI path and profiling as B1, with a JSON classifier."""
    def __init__(self, model_dir: Path, card: dict):
        self.model = NumpyTabularModel(json.loads(model_file_path(model_dir, card.get("model_file", "model.json")).read_text(encoding="utf-8")))
        self.cfg = SegConfig.from_dict(card.get("seg_config"))
        self.name = str(card.get("name") or "tabular_json")
        self.kind = "tabular_json"
        self._card = card
