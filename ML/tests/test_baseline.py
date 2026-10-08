"""ทดสอบ segment / features / linear_model / train_baseline ด้วยภาพสังเคราะห์"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from roastml.features import FEATURE_SETS, FEATURES_ALL, HIST_NAMES, border_features, image_features, pixel_stats, select
from roastml.linear_model import LinearSoftmax, Threshold3, spec_from_sklearn
from roastml.segment import SegConfig, segment
from tools import train_baseline as tb

RNG = np.random.default_rng(1234)


def beans_on_white(color=(90, 55, 35), n=12, size=(480, 640), bg=(235, 235, 230)):
    img = np.full((*size, 3), bg, np.uint8)
    h, w = size
    for _ in range(n):
        c = (int(RNG.integers(60, w - 60)), int(RNG.integers(60, h - 60)))
        cv2.ellipse(img, c, (24, 16), float(RNG.integers(0, 180)), 0, 360, color, -1)
    return img


def pile(color=(90, 55, 35), size=(400, 400)):
    base = np.array(color, np.float32)
    noise = RNG.normal(0, 12, (*size, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ segment
def test_segment_beans_on_white():
    img = beans_on_white()
    r = segment(img)
    assert r.mode == "beans" and r.wb_applied and r.n_regions >= 6
    x0, y0, x1, y1 = r.crop_box
    assert x1 - x0 < img.shape[1] and y1 - y0 <= img.shape[0]
    assert r.pixel_mask.sum() > 200


def test_segment_pile_and_full_fallback():
    assert segment(pile()).mode == "pile"
    cyan = np.zeros((300, 300, 3), np.uint8)
    cyan[..., 1] = cyan[..., 2] = RNG.integers(150, 255, (300, 300))  # ฟ้าอมเขียวสว่าง: ไม่มีพื้นหลัง ไม่เหมือนเมล็ด
    r = segment(cyan)
    assert r.mode == "full" and r.pixel_mask.any()  # ไม่มี "ไม่พบเมล็ด" — ใช้ภาพเต็ม


def test_segment_blank_white_falls_back_to_full():
    r = segment(np.full((300, 300, 3), 240, np.uint8))
    assert r.mode == "full" and r.pixel_mask.any()


def test_segment_find_beans_false_and_bad_input():
    assert segment(beans_on_white(), find_beans=False).mode == "pile"
    with pytest.raises(ValueError):
        segment(np.zeros((4, 4, 3), np.uint8))


# ------------------------------------------------------------ features
def test_pixel_stats_shape_and_hist():
    lab = np.stack([np.full(100, 40.0), np.full(100, 10.0), np.full(100, 20.0)], 1)
    x = pixel_stats(lab)
    assert x.shape == (len(FEATURES_ALL),)
    d = dict(zip(FEATURES_ALL, x))
    assert d["L_med"] == pytest.approx(40) and d["a_med"] == pytest.approx(10) and d["L_std"] == pytest.approx(0)
    assert sum(d[n] for n in HIST_NAMES) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pixel_stats(np.zeros((0, 3)))


def test_darker_beans_lower_L():
    light = image_features(beans_on_white((150, 105, 70))).x
    dark = image_features(beans_on_white((60, 35, 25))).x
    i = FEATURES_ALL.index("L_med")
    assert dark[i] < light[i]


def test_border_features_len():
    assert border_features(pile()).shape == (7,)


# ------------------------------------------------------------ linear_model
def test_linear_softmax_matches_sklearn():
    pytest.importorskip("sklearn")  # บน Pi ไม่มี sklearn → skip
    X = RNG.normal(size=(300, len(FEATURES_ALL)))
    y = np.array(["light", "medium", "dark"])[np.digitize(X[:, 0] + 0.3 * X[:, 7], [-0.5, 0.5])]
    names = FEATURE_SETS["Lab"]
    pipe = tb.make_pipe(1.0).fit(select(X, names), y)
    spec = spec_from_sklearn(pipe, names, ["light", "medium", "dark"])
    lm = LinearSoftmax(spec)
    np.testing.assert_allclose(lm.proba(X), pipe.predict_proba(select(X, names))[:, [list(pipe.classes_).index(c)
                                                                                     for c in lm.classes]], atol=1e-8)


def test_linear_softmax_rejects_bad_spec():
    spec = {"classes": ["a", "b", "c"], "feature_names": ["L_med"], "scaler_mean": [0], "scaler_scale": [0],
            "W": [[1], [2], [3]], "b": [0, 0, 0]}
    with pytest.raises(ValueError):
        LinearSoftmax(spec)


def test_threshold3():
    m = Threshold3(tb.b0_spec((20.0, 40.0)))
    X = np.zeros((3, len(FEATURES_ALL)))
    X[:, FEATURES_ALL.index("L_med")] = [10, 30, 50]
    assert list(np.array(m.classes)[m.proba(X).argmax(1)]) == ["dark", "medium", "light"]


# ------------------------------------------------------------ train_baseline
def test_metrics_cross_step():
    y = np.array(["light", "light", "dark", "medium"])
    p = np.array(["dark", "light", "light", "medium"])
    m = tb.metrics(y, p)
    assert m["n"] == 4 and m["acc"] == 0.5 and m["cross_step"] == 2 and m["cross_step_rate"] == 0.5
    assert m["per_class"]["medium"]["f1"] == 1.0


def test_metrics_macro_f1_ignores_absent_class():
    y = np.array(["dark", "dark", "medium"])
    m = tb.metrics(y, y)
    assert m["macro_f1"] == 1.0 and m["per_class"]["light"]["support"] == 0


def test_fit_b0_separable():
    v = np.array([10, 12, 14, 30, 32, 34, 50, 52, 54], float)
    y = np.array(["dark"] * 3 + ["medium"] * 3 + ["light"] * 3)
    t1, t2 = tb.fit_b0(v, y)
    assert 14 < t1 <= 30 and 34 < t2 <= 50


def test_loso_never_trains_on_heldout_source():
    src = np.array(["ontoum224"] * 4 + ["agtron"] * 4)
    X = np.zeros((8, len(FEATURES_ALL)))
    X[:, 0] = np.arange(8)
    y = np.array(["dark", "medium", "light", "dark"] * 2)
    seen = []

    def fit(Xt, yt):
        seen.append(set(Xt[:, 0].astype(int)))
        return tb.b0_spec((1.5, 2.5))

    tb.loso_predict(X, y, src, fit)
    assert seen == [{4, 5, 6, 7}, {0, 1, 2, 3}]  # ลำดับตาม FOLDS: ontoum224 ก่อน agtron


def test_load_rows_excludes_test_and_non_targets(tmp_path):
    cols = ["path", "label", "label_orig", "source", "group", "split", "license", "url", "md5", "phash", "roi",
            "device", "paper_path"]
    data = [("a", "dark", "trainval", "agtron"), ("b", "light", "test", "agtron"),
            ("c", "green", "trainval", "ontoum224"), ("d", "mixed", "trainval", "rf_boos"),
            ("e", "medium", "trainval", "ontoum224"), ("f", "dark", "trainval", "rf_hendi")]  # hendi ถูกตัด
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p, lab, sp, src in data:
            w.writerow({c: "" for c in cols} | {"path": p, "label": lab, "split": sp, "source": src})
    assert [r["path"] for r in tb.load_rows(tmp_path)] == ["a", "e"]


# ------------------------------------------------------------ main() end-to-end บนข้อมูลสังเคราะห์ (ไม่แตะข้อมูลจริง)
def _synthetic_dataset(root):
    """5 source × 3 คลาส × 4 ภาพ (กองเมล็ดสีตามระดับคั่ว) + แถว test ที่ต้องไม่ถูกอ่าน"""
    from PIL import Image

    from roastml.paths import ensure_layout

    root.mkdir(parents=True, exist_ok=True)
    ensure_layout(root)
    cols = ["path", "label", "label_orig", "source", "group", "split", "license", "url", "md5", "phash", "roi",
            "device", "paper_path"]
    color = {"light": (150, 105, 70), "medium": (105, 70, 45), "dark": (60, 38, 25)}
    rows = []
    for s in tb.FOLDS + list(tb.EXCLUDED_SOURCES):
        for lab, c in color.items():
            for i in range(4):
                p = f"raw/{s}/{lab}_{i}.jpg"
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                if s in tb.BOX_SOURCES:  # โครง YOLO: <split>/images + <split>/labels
                    p = f"raw/{s}/train/images/{lab}_{i}.jpg"
                    (root / p).parent.mkdir(parents=True, exist_ok=True)
                    lp = root / f"raw/{s}/train/labels/{lab}_{i}.txt"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    lp.write_text("0 0.5 0.5 0.6 0.6\n")
                Image.fromarray(pile(c, (96, 96))).save(root / p, quality=90)
                lo = f"agtron_{ {'light': 75, 'medium': 55, 'dark': 35}[lab] }" if s == "agtron" else lab
                rows.append({"path": p, "label": lab, "label_orig": lo, "source": s, "group": f"{s}:{i % 2}",
                             "split": "trainval", "md5": f"{s}{lab}{i}", "roi": "10 10 80 80" if s == "agtron" else ""})
    rows.append({"path": "raw/does_not_exist.jpg", "label": "dark", "source": "agtron", "split": "test",
                 "md5": "t", "roi": ""})  # ถ้าถูกอ่าน feature จะ error
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: "" for c in cols} | r)


def test_train_baseline_main_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")  # บน Pi ไม่มี sklearn → skip
    import json

    data = tmp_path / "data"
    _synthetic_dataset(data)
    monkeypatch.setattr(tb, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(tb, "MODEL_OUT", tmp_path / "models" / "current")
    assert tb.main(["--data-dir", str(data), "--workers", "1", "--export-model", "B1"]) == 0
    rep = json.loads((tmp_path / "results" / "baseline_loso.json").read_text(encoding="utf-8"))
    assert rep["n_errors"] == 0 and rep["n_images"] == 12 * len(tb.FOLDS)  # ไม่โหลดแถว test และ rf_hendi
    assert rep["n_box_rows_without_bbox"] == 0
    assert set(rep["by_run"]) == set(tb.RUNS)
    for view in ("bbox", "pipeline"):
        f = rep["by_run"]["R1"]["B0"]["folds"][view]
        assert set(f) >= set(tb.FOLDS) and "rf_devlong" not in f and "rf_hendi" not in f
    assert {"by_run", "decision", "selected", "shortcut_probe", "wb_guard"} <= set(rep)
    assert rep["by_run"]["R2"]["B1"]["agtron_by_value"]  # แยกตามค่า Agtron
    assert rep["wb_guard"]["rf_robusta"]["n"] == 12 and "box_wb_not_applied_pct" in rep["wb_guard"]["rf_robusta"]
    assert isinstance(rep["decision"]["r1_kept"], bool)
    with open(tmp_path / "results" / "baseline_loso.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["eval_view"] for r in rows} == {"bbox", "pipeline"} and {r["run"] for r in rows} == set(tb.RUNS)
    assert not any(r["fold"] == "rf_hendi" for r in rows)
    card = json.loads((tmp_path / "models" / "current" / "model_card.json").read_text(encoding="utf-8"))
    assert card["backend"] == "b1_linear" and card["feature_set"] == "Lab_hist"  # --export-model B1
    assert card["low_conf_threshold"] == tb.LOW_CONF_THRESHOLD and "rf_hendi" not in card["loso_folds"]
    assert rep["selected"]["selected_by"].endswith("B1") and rep["selected"]["run"] in ("R1", "R2")
    assert (tmp_path / "results" / "baseline_confusion" / "R1_B1_pipeline_agtron.csv").is_file()


def test_video_cap_mask():
    groups = np.array(["s:video:A"] * 25 + ["s:video:B"] * 4 + ["s:file:x"] * 3)
    paths = np.array([f"p{i:03d}" for i in range(len(groups))])
    m = tb.video_cap_mask(groups, paths, cap=10, seed=1)
    assert m[:25].sum() == 10 and m[25:29].all() and m[29:].all()
    np.testing.assert_array_equal(m, tb.video_cap_mask(groups, paths, cap=10, seed=1))  # ทำซ้ำได้
    # ลำดับแถวไม่มีผล (เรียงตาม path ก่อนสุ่ม)
    perm = np.random.default_rng(0).permutation(len(groups))
    m2 = tb.video_cap_mask(groups[perm], paths[perm], cap=10, seed=1)
    assert set(paths[perm][m2]) == set(paths[m])


def test_loso_train_mask_limits_training_rows():
    src = np.array(["ontoum224"] * 4 + ["agtron"] * 4)
    X = np.zeros((8, len(FEATURES_ALL)))
    X[:, 0] = np.arange(8)
    y = np.array(["dark", "medium", "light", "dark"] * 2)
    seen = []

    def fit(Xt, yt):
        seen.append(set(Xt[:, 0].astype(int)))
        return tb.b0_spec((1.5, 2.5))

    tb.loso_predict(X, y, src, fit, train_mask=np.array([True, False, True, True, True, False, True, True]))
    assert seen == [{4, 6, 7}, {0, 2, 3}]


# ------------------------------------------------------------ โหมด full (แก้ 8 ต.ค.): เลือกพิกเซลที่ห่างจากพื้นหลัง
def test_full_mode_picks_beans_not_paper():
    img = np.full((480, 640, 3), (235, 235, 230), np.uint8)
    beans = np.zeros(img.shape[:2], np.uint8)
    for c in ((200, 200), (320, 260), (450, 220)):
        cv2.ellipse(img, c, (30, 20), 20, 0, 360, (70, 45, 30), -1)
        cv2.ellipse(beans, c, (30, 20), 20, 0, 360, 1, -1)
    # บังคับให้กรองรูปทรงไม่ผ่าน → โหมด full (แบบที่เกิดกับ rf_boos)
    cfg = SegConfig(min_solidity=1.01, cluster_min_solidity=1.01)
    r = segment(img, cfg)
    assert r.mode == "full" and "full_trimmed" not in r.notes
    inside = (r.pixel_mask & beans.astype(bool)).sum() / r.pixel_mask.sum()
    assert inside > 0.9  # เดิม (ตัด percentile) ได้กระดาษเกือบทั้งหมด


def test_full_mode_blank_image_falls_back_to_trim():
    r = segment(np.full((300, 300, 3), 240, np.uint8))
    assert r.mode == "full" and "full_trimmed" in r.notes and r.pixel_mask.any()


# ------------------------------------------------------------ box_features ใช้ WB ตัวเดียวกับ segment (8 ต.ค.)
def _bean_on_paper(paper, bean=(95, 62, 40), size=(480, 640), seed=3):
    """เมล็ดเดียวขนาดใหญ่กลางภาพบนกระดาษ · คืน (ภาพ, bbox สัมพัทธ์ cx, cy, w, h)"""
    r = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), paper, np.float32) + r.normal(0, 2, (h, w, 3))
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (110, 75), 0, 0, 360, 1, -1)
    beanpx = np.array(bean, np.float32) + r.normal(0, 8, (h, w, 3))
    img[m.astype(bool)] = beanpx[m.astype(bool)]
    return np.clip(img, 0, 255).astype(np.uint8), [(0.5, 0.5, 220 / w, 150 / h)]


def test_box_features_matches_smart_crop_when_crop_equals_bbox():
    from roastml.features import box_features

    img, boxes = _bean_on_paper((205, 195, 175))  # กระดาษอมเหลือง/ทึม: ผ่าน guard แต่ต้อง WB
    pipe = image_features(img)
    assert pipe.seg.mode == "beans" and pipe.seg.wb_applied
    bx = box_features(img, boxes)
    assert bx.wb_applied
    names = ["L_med", "a_med", "b_med"]
    d = {n: abs(select(pipe.x, [n])[0] - select(bx.x, [n])[0]) for n in names}
    assert all(v < 2.0 for v in d.values()), d  # tolerance 2 หน่วย Lab
    # ถ้าไม่ WB ใน box_features (พฤติกรรมเดิม) ต่างจาก pipeline เกิน tolerance → test นี้มีความหมาย
    raw = box_features(img, boxes, white_balance=False)
    assert abs(select(pipe.x, ["L_med"])[0] - select(raw.x, ["L_med"])[0]) > 2.0


def test_box_features_no_wb_on_colored_background():
    from roastml.features import box_features

    img, boxes = _bean_on_paper((60, 160, 60))  # พื้นเขียว: chroma สูง ไม่ผ่าน guard
    bx = box_features(img, boxes)
    assert not bx.wb_applied
    np.testing.assert_allclose(bx.x, box_features(img, boxes, white_balance=False).x)
    assert not image_features(img).seg.wb_applied  # pipeline ก็ไม่ WB เหมือนกัน
