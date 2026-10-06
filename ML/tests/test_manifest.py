"""ทดสอบ tools/agtron.py + tools/make_manifest.py ด้วยข้อมูลสังเคราะห์"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from roastml.paths import ensure_layout
from tools import agtron as ag
from tools import make_manifest as mm


# ------------------------------------------------------------ agtron
def test_label_map_is_fixed():
    assert ag.LABEL_MAP == {75: "light", 65: "medium", 55: "medium", 45: "dark", 35: "dark", 25: "dark"}


def test_choose_test_devices_deterministic():
    ids = [str(i) for i in range(7)]
    a = ag.choose_test_devices(ids)
    assert a == ag.choose_test_devices(list(reversed(ids)))  # ไม่ขึ้นกับลำดับ input
    assert len(a) == 2 and a == sorted(a, key=int)
    assert a == ["0", "3"]  # ค่าที่บันทึกใน split_decisions.md (seed 20261006)


def _exif_img(path, size, orientation):
    """ภาพที่ครึ่งซ้าย (ในพิกัดดิบ) เป็นสีแดง ครึ่งขวาเป็นน้ำเงิน + EXIF orientation"""
    w, h = size
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, : w // 2] = (255, 0, 0)
    arr[:, w // 2:] = (0, 0, 255)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = orientation
    im.save(path, exif=exif)


def test_load_roi_uses_exif_transposed_coords(tmp_path):
    p = tmp_path / "a.jpg"
    _exif_img(p, (80, 40), 6)  # หมุน 90° → ภาพหลังหมุนสูง 80 กว้าง 40 · แดงอยู่ "บน"
    top = np.asarray(ag.load_roi(p, (5, 5, 35, 35)), dtype=int).mean(axis=(0, 1))
    bottom = np.asarray(ag.load_roi(p, (5, 45, 35, 75)), dtype=int).mean(axis=(0, 1))
    assert top[0] > 200 and top[2] < 60      # แดง
    assert bottom[2] > 200 and bottom[0] < 60  # น้ำเงิน
    # พิกัดที่ใช้ได้กับภาพดิบ (80x40) แต่หลุดภาพหลังหมุน (40x80) ต้อง error
    with pytest.raises(ValueError):
        ag.load_roi(p, (0, 0, 70, 30))


def write_agtron(base, rows, devices=(("0", "PhoneA"), ("1", "PhoneB"))):
    img_dir = base / "RAW" / "RAW"
    img_dir.mkdir(parents=True)
    (base / "devices.csv").write_text("device_id;name;aperture;height;width\n"
                                      + "".join(f"{d};{n};1.8;64;64\n" for d, n in devices))
    lines = ["device_id;name_coffee;name_paper;agtron;flash;X1;Y1;X2;Y2;H"]
    for i, (dev, name, paper, a) in enumerate(rows):
        lines.append(f"{dev};{name};{paper};{a};0;8;8;56;56;48")
        for n in (name, paper):
            if not (img_dir / n).exists():
                Image.fromarray(np.full((64, 64, 3), 40 + 20 * i, np.uint8)).save(img_dir / n)
    (base / "photos.csv").write_text("\n".join(lines) + "\n")


def test_read_photos_dup_rows(tmp_path):
    base = tmp_path / "agtron"
    write_agtron(base, [("0", "c1.jpg", "p.jpg", 25), ("0", "c1.jpg", "p.jpg", 25),
                        ("1", "c2.jpg", "p.jpg", 75), ("1", "c2.jpg", "p.jpg", 65), ("1", "c3.jpg", "p.jpg", 55)])
    photos, rep = ag.read_photos(base)
    assert [p.name_coffee for p in photos] == ["c1.jpg", "c3.jpg"]
    assert rep["dup_rows_identical"] == 1 and rep["dup_rows_conflicting_dropped"] == 2


# ------------------------------------------------------------ label / group
@pytest.mark.parametrize("source,orig,expect", [
    ("ontoum224", "Green", ("green", "")),
    ("rf_hendi", "Raw", ("green", "")),
    ("rf_hendi", "Empty", ("empty", "")),
    ("rf_devlong", "Light", ("light", "")),
    ("rf_devlong", "Yellow", (None, "unknown_class:Yellow")),
    ("rf_robusta", "Maw", (None, "maw_only")),
    ("rf_robusta", "Dark Roast|Maw", ("mixed", "")),
    ("rf_robusta", "Dark Roast", ("dark", "")),
    ("rf_robusta", "Raw", ("green", "")),
    ("rf_robusta", "Dark Roast|Raw", ("mixed", "")),
    ("rf_robusta", "(no_label)", (None, "yolo_no_label")),
])
def test_map_label(source, orig, expect):
    assert mm.map_label(source, orig) == expect


@pytest.mark.parametrize("source,stem,expect", [
    ("rf_hendi", "20240817_103653_001_saved_jpg", "rf_hendi:20240817_103653"),
    ("rf_boos", "WhatsApp-Video-2026-05-19-at-9_14_47-PM_mp4-12_jpg", "rf_boos:video:WhatsApp-Video-2026-05-19-at-9_14_47-PM"),
    ("rf_robusta", "Coffee-Test_mp4-606_jpg", "rf_robusta:video:Coffee-Test"),
    ("rf_robusta", "LINE_ALBUM_220329_14_0_jpg", "rf_robusta:LINE_ALBUM_220329"),
    ("rf_robusta", "4620693218556181757-50128f8d82874bfcfd523ae927ae3180-22031110_JPG", "rf_robusta:msg:22031110"),
    ("rf_robusta", "DSC0001_JPG", "rf_robusta:file:x/y.jpg"),
    ("rf_devlong", "light-1-_png", "rf_devlong:file:x/y.jpg"),
])
def test_group_of(source, stem, expect):
    assert mm.group_of(source, stem, "x/y.jpg") == expect


# ------------------------------------------------------------ dedupe
def _row(path, source, label, split_orig, md5):
    return mm.Row(path, label, label, source, "g", "trainval", "", "", md5=md5, split_orig=split_orig)


def test_dedupe_md5_priority_and_conflict():
    rows = [
        _row("o/test/a.png", "ontoum224", "dark", "test", "m1"),
        _row("o/train/a.png", "ontoum224", "dark", "train", "m1"),
        _row("h/x.jpg", "rf_hendi", "light", "train", "m2"),
        _row("o/train/b.png", "ontoum224", "light", "valid", "m2"),  # ข้าม source → ontoum224 ชนะ
        _row("h/c1.jpg", "rf_hendi", "light", "train", "m3"),
        _row("h/c2.jpg", "rf_hendi", "dark", "train", "m3"),  # label ขัดกัน → ตัดทั้งคู่
        _row("h/u.jpg", "rf_hendi", "medium", "train", "m4"),
    ]
    dropped = []
    keep, stats = mm.dedupe_md5(rows, dropped)
    assert sorted(r.path for r in keep) == ["h/u.jpg", "o/train/a.png", "o/train/b.png"]
    assert stats["groups_label_conflict_dropped"] == 1 and stats["groups_cross_source"] == 1
    assert len(dropped) == 4


# ------------------------------------------------------------ end-to-end สังเคราะห์
def test_build_end_to_end(tmp_path):
    ensure_layout(tmp_path)
    raw = tmp_path / "raw"

    def img(p, v):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), v, np.uint8)).save(p)

    img(raw / "ontoum224" / "train" / "Dark" / "d1.png", 10)
    img(raw / "ontoum224" / "test" / "Dark" / "d9.png", 10)  # md5 ซ้ำ train → ตัด
    img(raw / "ontoum224" / "train" / "Green" / "g1.png", 200)
    img(raw / "rf_hendi" / "train" / "Light" / "20240101_000000_001_jpg.rf.a.jpg", 150)
    img(raw / "rf_devlong" / "train" / "Medium" / "m-1-_png.rf.b.jpg", 100)
    for s, names in (("rf_robusta", "['Dark Roast', 'Maw']"), ("rf_boos", "['Dark Roast', 'Light Roast']")):
        (raw / s).mkdir(parents=True)
        (raw / s / "data.yaml").write_text(f"names: {names}\n")
        for i, lab in enumerate(["0\n", "1\n", "0\n1\n"]):
            img(raw / s / "train" / "images" / f"v_mp4-{i}_jpg.rf.{i}.jpg", 30 + 40 * i + (5 if s == "rf_boos" else 0))
            (raw / s / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (raw / s / "train" / "labels" / f"v_mp4-{i}_jpg.rf.{i}.txt").write_text(lab.replace("\n", " .5 .5 .1 .1\n"))
    # ทุกเครื่องมีภาพ (เหมือนข้อมูลจริง) → สุ่มได้ 0 กับ 3
    others = [(d, f"c{d}x.jpg", f"p{d}.jpg", 35) for d in ("2", "4", "5", "6")]
    write_agtron(raw / "agtron", [("0", "c1.jpg", "p0.jpg", 75), ("3", "c2.jpg", "p3.jpg", 45), ("1", "c3.jpg", "p1.jpg", 55)] + others,
                 devices=(("0", "PhoneA"), ("1", "PhoneB"), ("2", "C"), ("3", "D"), ("4", "E"), ("5", "F"), ("6", "G")))

    rows, dropped, extra = mm.build(tmp_path)
    by = {(r.source, r.path.split("/")[-1]): r for r in rows}
    assert ("ontoum224", "d9.png") not in by and by[("ontoum224", "d1.png")].label == "dark"
    assert by[("ontoum224", "g1.png")].label == "green"
    assert by[("rf_robusta", "v_mp4-0_jpg.rf.0.jpg")].label == "dark"
    assert ("rf_robusta", "v_mp4-1_jpg.rf.1.jpg") not in by  # Maw อย่างเดียว
    assert by[("rf_robusta", "v_mp4-2_jpg.rf.2.jpg")].label == "mixed"
    assert by[("rf_boos", "v_mp4-2_jpg.rf.2.jpg")].label == "mixed"
    a1, a2, a3 = by[("agtron", "c1.jpg")], by[("agtron", "c2.jpg")], by[("agtron", "c3.jpg")]
    assert (a1.label, a1.split) == ("light", "test") and (a2.label, a2.split) == ("dark", "test")
    assert (a3.label, a3.split) == ("medium", "trainval")
    assert a1.roi == "8 8 56 56" and a1.paper_path.endswith("p0.jpg")
    assert all(r.split == "trainval" for r in rows if r.source != "agtron")
    assert all(len(r.phash) == 16 and len(r.md5) == 32 for r in rows)
    reasons = {d["reason"].split(":")[0] for d in dropped}
    assert {"md5_dup_of", "maw_only"} <= reasons
    assert extra["agtron"]["test_devices"] == ["0:PhoneA", "3:D"]

    # เขียนไฟล์ได้ และคอลัมน์ตรง spec + คอลัมน์เสริม
    out = tmp_path / "manifest.csv"
    mm.write_csv_atomic(out, [vars(r) for r in rows], mm.COLUMNS)
    with open(out, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header[:10] == ["path", "label", "label_orig", "source", "group", "split", "license", "url", "md5", "phash"]
    assert "split_orig" not in header
