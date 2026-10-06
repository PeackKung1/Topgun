"""ทดสอบ tools ข้อมูล (imghash, index_sources, dup_stems, explore_agtron) ด้วยไฟล์สังเคราะห์ใน tmp_path"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tools import dup_stems as ds
from tools import explore_agtron as ea
from tools import index_sources as ix
from tools.imghash import hamming, hamming_matrix, load_gray, md5_file, phash_array

RNG = np.random.default_rng(1234)


def save_img(path, arr, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, **kw)
    return path


def textured(seed: int, size: int = 128) -> np.ndarray:
    """ภาพมีโครงสร้าง (ไม่ใช่ noise ล้วน) ให้ pHash มีความหมาย"""
    r = np.random.default_rng(seed)
    small = r.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((size, size), Image.BILINEAR))


# ------------------------------------------------------------ imghash
def test_phash_same_image_reencoded_and_resized(tmp_path):
    a = save_img(tmp_path / "a.png", textured(1))
    b = tmp_path / "b.jpg"
    Image.open(a).resize((300, 300)).save(b, quality=60)
    c = save_img(tmp_path / "c.png", textured(2))
    ha, hb, hc = (phash_array(load_gray(p)) for p in (a, b, c))
    assert hamming(ha, hb) <= 4
    assert hamming(ha, hc) > 10
    assert md5_file(a) != md5_file(b)


def test_hamming_matrix_matches_scalar():
    hs = RNG.integers(0, 2**63, 6, dtype=np.uint64)
    m = hamming_matrix(hs)
    for i in range(6):
        for j in range(6):
            assert m[i, j] == hamming(int(hs[i]), int(hs[j]))


# ------------------------------------------------------------ index_sources
@pytest.mark.parametrize("name,stem", [
    ("abc_jpg.rf.0f3e9a.jpg", "abc_jpg"),
    ("20240817_103645_001_jpg.rf.0537b970fa48e04a0c327470361b05aa.jpg", "20240817_103645_001_jpg"),
    ("dark (1).png", "dark (1)"),
    ("x.RF.ABC.jpg", "x"),
])
def test_rf_stem(name, stem):
    assert ix.rf_stem(name) == stem


def test_index_yolo_source(tmp_path):
    base = tmp_path / "raw" / "rf_boos"
    base.mkdir(parents=True)
    (base / "data.yaml").write_text("nc: 2\nnames: ['Dark', 'Light']\n", encoding="utf-8")
    for split, name, lines in [("train", "a_jpg.rf.1", "0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n"),
                               ("valid", "b_jpg.rf.2", "1 .5 .5 .1 .1\n"),
                               ("valid", "c_jpg.rf.3", "")]:
        save_img(base / split / "images" / f"{name}.jpg", textured(3, 32))
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        (base / split / "labels" / f"{name}.txt").write_text(lines)
    items = {i.stem: i for i in ix.index_source(tmp_path, "rf_boos")}
    assert items["a_jpg"].label_orig == "Dark|Light" and items["a_jpg"].n_boxes == 2
    assert items["b_jpg"].label_orig == "Light" and items["b_jpg"].split_orig == "valid"
    assert items["c_jpg"].label_orig == "(no_label)"
    assert items["a_jpg"].path == "raw/rf_boos/train/images/a_jpg.rf.1.jpg"
    assert (items["a_jpg"].width, items["a_jpg"].height) == (32, 32)


# ------------------------------------------------------------ dup_stems
@pytest.fixture
def dup_tree(tmp_path):
    """rf_hendi สังเคราะห์:
    s1: ไฟล์เดียวกัน (byte ตรง) ใน train + valid                         → (ก)
    s2: ไฟล์เดียวกัน 2 ไฟล์ใน train คลาสเดียว                           → (ก2)
    s3: ภาพเดียวกัน (re-encode) คนละคลาส                                  → (ค)
    s4: คนละภาพ ชื่อเดียวกัน คนละคลาส                                     → (ข) ไม่ใช่ label ขัดกัน
    u1/u2: ไม่ซ้ำ stem แต่ byte ตรงกันและคนละคลาส                        → md5 conflict ข้าม stem
    """
    b = tmp_path / "raw" / "rf_hendi"
    img1 = save_img(b / "train" / "Light" / "s1_jpg.rf.aa.jpg", textured(10))
    (b / "valid" / "Light").mkdir(parents=True)
    (b / "valid" / "Light" / "s1_jpg.rf.bb.jpg").write_bytes(img1.read_bytes())
    img2 = save_img(b / "train" / "Dark" / "s2_jpg.rf.aa.jpg", textured(20))
    (b / "train" / "Dark" / "s2_jpg.rf.bb.jpg").write_bytes(img2.read_bytes())
    save_img(b / "train" / "Light" / "s3_jpg.rf.aa.jpg", textured(30))
    save_img(b / "train" / "Medium" / "s3_jpg.rf.bb.jpg", textured(30), quality=50)
    save_img(b / "train" / "Light" / "s4_jpg.rf.aa.jpg", textured(40))
    save_img(b / "valid" / "Dark" / "s4_jpg.rf.bb.jpg", textured(41))
    u = save_img(b / "train" / "Light" / "u1_jpg.rf.aa.jpg", textured(50))
    (b / "train" / "Dark" / "u2_jpg.rf.aa.jpg").write_bytes(u.read_bytes())
    return tmp_path


def test_dup_stems_categories(dup_tree):
    hashed = ds.hash_items(dup_tree, ix.index_source(dup_tree, "rf_hendi"))
    summary, pairs, clusters = ds.analyze(hashed, threshold=6)
    assert summary["n_stem_groups_dup"] == 4
    assert summary["clusters_by_category"] == {"ก_cross_split": 1, "ก2_same_split": 1, "ค_label_conflict": 1}
    assert summary["ข_stem_groups_with_different_images"] == 1
    assert summary["ข_pairs_different_image_diff_label"] == 1
    conflict = {r["stem"] for r in clusters if r["category"] == "ค_label_conflict"}
    assert conflict == {"s3_jpg"}
    ident = {r["stem"]: r["identity"] for r in pairs}
    assert ident == {"s1_jpg": "md5", "s2_jpg": "md5", "s3_jpg": "phash", "s4_jpg": "different"}

    md5_sum, md5_rows = ds.global_md5_dups(hashed)
    assert md5_sum["md5_groups_other_stem"] == 1 and md5_sum["groups_diff_label"] == 1
    assert {r["path"].split("/")[-1] for r in md5_rows} == {"u1_jpg.rf.aa.jpg", "u2_jpg.rf.aa.jpg"}


def test_dup_stems_threshold_zero_makes_reencode_different(dup_tree):
    hashed = ds.hash_items(dup_tree, ix.index_source(dup_tree, "rf_hendi"))
    # ถ้า pHash ของ re-encode ไม่ตรงเป๊ะ ที่ T=-1 ต้องไม่นับเป็นภาพเดียวกัน (md5 ยังนับ)
    summary, _, _ = ds.analyze(hashed, threshold=-1)
    assert summary["clusters_by_category"]["ค_label_conflict"] == 0
    assert summary["clusters_by_category"]["ก_cross_split"] == 1


# ------------------------------------------------------------ explore_agtron
def test_explore_agtron_synthetic(tmp_path):
    base = tmp_path / "agtron"
    img_dir = base / "RAW" / "RAW"
    for n in ("c1.jpg", "c2.jpg", "c3.jpg", "p1.jpg", "p2.jpg", "extra.jpg"):
        save_img(img_dir / n, textured(5, 40))
    (base / "devices.csv").write_text("device_id;name;aperture;height;width\n0;PhoneA;1.8;40;40\n1;PhoneB;2.0;40;40\n")
    (base / "photos.csv").write_text(
        "device_id;name_coffee;name_paper;agtron;flash;X1;Y1;X2;Y2;H\n"
        "0;c1.jpg;p1.jpg;25;0;0;0;10;10;5\n"
        "0;c2.jpg;p1.jpg;35;1;0;0;10;10;5\n"
        "1;c3.jpg;p2.jpg;35;0;0;0;12;12;6\n"
    )
    r = ea.explore(base)
    assert r["n_image_files"] == 6 and r["files_not_in_csv"] == 1
    assert r["agtron_values"] == {25: 1, 35: 2}
    assert r["coffee_images_device_x_agtron"] == {"PhoneA": {25: 1, 35: 1}, "PhoneB": {35: 1}}
    pg = r["paper_groups"]
    assert pg["n_groups"] == 2 and pg["groups_with_more_than_one_agtron"] == 1
    assert pg["groups_per_device_x_agtron"] == {"PhoneA": {25: 1, 35: 1}, "PhoneB": {35: 1}}
    assert r["image_sizes_by_device"] == {"PhoneA": {"40x40": 2}, "PhoneB": {"40x40": 1}}
