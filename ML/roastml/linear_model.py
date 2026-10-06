"""inference ของ B0 (threshold บน L_med) และ B1 (StandardScaler + multinomial LogisticRegression) ด้วย numpy ล้วน

ไม่ใช้ pickle ของ sklearn: ผูก version, Pi ไม่ต้องลง sklearn, model.json อ่านด้วยตาได้ ขนาดไม่กี่ KB
spec (dict ใน model.json):
  B0: {"type": "threshold", "feature": "L_med", "classes": ["dark","medium","light"], "thresholds": [t1, t2]}
  B1: {"type": "softmax", "classes": [...], "feature_names": [...], "scaler_mean", "scaler_scale", "W", "b"}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURES_ALL


class LinearSoftmax:
    def __init__(self, spec: dict):
        if spec.get("type", "softmax") != "softmax":
            raise ValueError(f"spec type ไม่ใช่ softmax: {spec.get('type')}")
        self.spec = spec
        self.classes: list[str] = list(spec["classes"])
        self.feature_names: list[str] = list(spec["feature_names"])
        self.idx = np.array([FEATURES_ALL.index(n) for n in self.feature_names])
        self.mean = np.asarray(spec["scaler_mean"], np.float64)
        self.scale = np.asarray(spec["scaler_scale"], np.float64)
        self.W = np.asarray(spec["W"], np.float64)  # (K, D)
        self.b = np.asarray(spec["b"], np.float64)  # (K,)
        d = len(self.feature_names)
        if self.mean.shape != (d,) or self.scale.shape != (d,) or self.W.shape != (len(self.classes), d) \
                or self.b.shape != (len(self.classes),):
            raise ValueError("ขนาด array ใน spec ไม่ตรงกัน")
        if np.any(self.scale <= 0):
            raise ValueError("scaler_scale ต้อง > 0")

    @classmethod
    def load(cls, path: str | Path) -> "LinearSoftmax":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def proba(self, X_all: np.ndarray) -> np.ndarray:
        """X_all (n, len(FEATURES_ALL)) → (n, K)"""
        X_all = np.atleast_2d(np.asarray(X_all, np.float64))
        z = (X_all[:, self.idx] - self.mean) / self.scale
        logits = z @ self.W.T + self.b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


class Threshold3:
    """B0: L_med < t1 → classes[0] · t1 ≤ L_med < t2 → classes[1] · ≥ t2 → classes[2] (prob = one-hot)"""

    def __init__(self, spec: dict):
        self.classes = list(spec["classes"])
        self.feature = spec.get("feature", "L_med")
        self.t1, self.t2 = (float(v) for v in spec["thresholds"])
        if len(self.classes) != 3 or not self.t1 <= self.t2:
            raise ValueError("B0 ต้องมี 3 คลาสและ t1 ≤ t2")
        self.i = FEATURES_ALL.index(self.feature)

    def predict_idx(self, X_all: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(np.asarray(X_all, np.float64))[:, self.i]
        return np.where(v < self.t1, 0, np.where(v < self.t2, 1, 2))

    def proba(self, X_all: np.ndarray) -> np.ndarray:
        idx = self.predict_idx(X_all)
        out = np.zeros((len(idx), 3))
        out[np.arange(len(idx)), idx] = 1.0
        return out


def spec_from_sklearn(pipe, feature_names: list[str], classes: list[str]) -> dict:
    """sklearn Pipeline(StandardScaler, LogisticRegression) → spec dict (เรียกบน notebook เท่านั้น)"""
    scaler, clf = pipe.steps[0][1], pipe.steps[-1][1]
    order = [list(clf.classes_).index(c) for c in classes]  # เรียงแถว W ตาม classes ที่ต้องการ
    W, b = np.asarray(clf.coef_), np.asarray(clf.intercept_)
    if W.shape[0] == 1:
        raise ValueError("ต้องเป็น multinomial ≥ 3 คลาส")
    return {
        "type": "softmax",
        "classes": list(classes),
        "feature_names": list(feature_names),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "W": W[order].tolist(),
        "b": b[order].tolist(),
    }
