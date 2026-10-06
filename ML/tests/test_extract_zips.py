"""ทดสอบ tools/extract_zips.py ด้วย zip สังเคราะห์ใน tmp_path"""

from __future__ import annotations

import zipfile

import pytest

from roastml.paths import ensure_layout
from tools import extract_zips as ez


@pytest.fixture
def root(tmp_path):
    ensure_layout(tmp_path)
    return tmp_path


def make_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def snapshot(d):
    """{relpath: bytes} ของทุกไฟล์ใต้ d — ใช้เช็คว่าไม่มีอะไรถูกแตะ"""
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def no_tmp_left(root):
    tmp = root / "cache" / ez.TMP_PARENT
    return not tmp.exists() or not any(tmp.iterdir())


# ---------------------------------------------------------------- ปกติ
def test_extract_ok(root):
    z = make_zip(root / "zips" / "rf_hendi.zip", {
        "train/Light/a.jpg": b"1", "train/Dark/b.jpg": b"2", "README.txt": b"r",
    })
    zip_bytes = z.read_bytes()
    r = ez.extract_one(z, root)
    assert r.status == "extracted", r.message
    assert snapshot(root / "raw" / "rf_hendi") == {
        "README.txt": b"r", "train/Dark/b.jpg": b"2", "train/Light/a.jpg": b"1",
    }
    assert z.read_bytes() == zip_bytes  # zip ไม่ถูกลบ/แก้
    assert no_tmp_left(root)


def test_unknown_source_skipped(root):
    z = make_zip(root / "zips" / "random.zip", {"a.jpg": b"1"})
    r = ez.extract_one(z, root)
    assert r.status == "unknown_source"
    assert not (root / "raw" / "random").exists()


# ---------------------------------------------------------------- ปลายทางมีอยู่แล้ว
def test_existing_dest_untouched(root):
    dest = root / "raw" / "agtron"
    dest.mkdir()
    (dest / "old.jpg").write_bytes(b"old")
    z = make_zip(root / "zips" / "agtron.zip", {"old.jpg": b"NEW", "extra.jpg": b"x"})
    r = ez.extract_one(z, root)
    assert r.status == "exists"
    assert snapshot(dest) == {"old.jpg": b"old"}


def test_existing_dest_as_file_untouched(root):
    f = root / "raw" / "agtron"
    f.write_bytes(b"file")
    z = make_zip(root / "zips" / "agtron.zip", {"a.jpg": b"1"})
    assert ez.extract_one(z, root).status == "exists"
    assert f.read_bytes() == b"file"


# ---------------------------------------------------------------- zip-slip
@pytest.mark.parametrize("evil", [
    "../evil.jpg",
    "ok/../../evil.jpg",
    "..\\evil.jpg",
    "/abs/evil.jpg",
    "\\abs\\evil.jpg",
    "C:/evil.jpg",
    "C:evil.jpg",
    "a/file.jpg:stream",
])
def test_zip_slip_rejects_whole_file(root, evil):
    z = make_zip(root / "zips" / "rf_color.zip", {"good/a.jpg": b"1", evil: b"x"})
    r = ez.extract_one(z, root)
    assert r.status == "unsafe", (evil, r.message)
    assert not (root / "raw" / "rf_color").exists()  # ไม่แตกแม้แต่ไฟล์ที่ดี
    assert not (root / "evil.jpg").exists() and not (root / "raw" / "evil.jpg").exists()
    assert no_tmp_left(root)


def test_case_duplicate_names_warn(root):
    z = make_zip(root / "zips" / "rf_boos.zip", {"A.jpg": b"1", "a.jpg": b"2"})
    r = ez.extract_one(z, root)
    assert r.status == "extracted" and r.warnings


# ---------------------------------------------------------------- zip เสีย
def test_not_a_zip(root):
    z = root / "zips" / "rf_boos.zip"
    z.write_bytes(b"this is not a zip file at all")
    r = ez.extract_one(z, root)
    assert r.status == "bad_zip"
    assert z.read_bytes() == b"this is not a zip file at all"  # ไม่ลบ
    assert not (root / "raw" / "rf_boos").exists()


def test_truncated_zip(root):
    z = make_zip(root / "zips" / "rf_devlong.zip", {f"c/{i}.jpg": bytes(200) for i in range(5)})
    data = z.read_bytes()
    z.write_bytes(data[: len(data) // 2])
    r = ez.extract_one(z, root)
    assert r.status == "bad_zip"
    assert not (root / "raw" / "rf_devlong").exists()


def test_crc_error_leaves_nothing_half_done(root):
    payload = b"ABCDEFGH" * 64
    z = make_zip(root / "zips" / "ontoum224.zip", {"ok.jpg": b"fine", "bad.jpg": payload})
    data = bytearray(z.read_bytes())
    i = data.index(payload)  # ZIP_STORED → payload อยู่ตรงๆ ในไฟล์
    data[i] ^= 0xFF  # ทำให้ CRC ไม่ตรง
    z.write_bytes(bytes(data))
    r = ez.extract_one(z, root)
    assert r.status == "bad_zip", r.message
    assert not (root / "raw" / "ontoum224").exists()
    assert no_tmp_left(root)


# ---------------------------------------------------------------- extract_all + CLI + survey
def test_extract_all_mixed_and_cli_exit_code(root, capsys):
    make_zip(root / "zips" / "rf_hendi.zip", {"train/Light/a.jpg": b"1", "train/Light/b.png": b"2"})
    (root / "zips" / "rf_boos.zip").write_bytes(b"broken")
    make_zip(root / "zips" / "foo.zip", {"a.jpg": b"1"})
    code = ez.main(["--data-dir", str(root)])
    out = capsys.readouterr().out
    assert code == 1  # มี bad_zip
    assert "[EXTRACTED] rf_hendi" in out and "[BAD_ZIP] rf_boos" in out and "[WARN] foo" in out
    assert "=== rf_hendi: 2 ภาพ" in out


def test_survey_counts(root):
    src = root / "raw" / "rf_robusta"
    (src / "train" / "Light").mkdir(parents=True)
    (src / "valid" / "Dark").mkdir(parents=True)
    (src / "train" / "Light" / "a.JPG").write_bytes(b"12345")
    (src / "valid" / "Dark" / "b.png").write_bytes(b"1")
    (src / "data.yaml").write_bytes(b"y")
    s = ez.survey_source(src)
    assert (s.n_files, s.n_images, s.total_bytes) == (3, 2, 7)
    assert s.image_dirs == {"train/Light": 1, "valid/Dark": 1}
    assert "train/  (1 ภาพ)" in s.tree and "  Light/  (1 ภาพ)" in s.tree
