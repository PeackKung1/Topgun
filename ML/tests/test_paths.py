"""ทดสอบ roastml.paths: ลำดับ env → local_paths.json → error (ไม่ fallback เงียบๆ)"""

from __future__ import annotations

import json

import pytest

from roastml.paths import ENV_VAR, SUBDIRS, DataDirError, data_dir, ensure_layout


@pytest.fixture
def dirs(tmp_path):
    a, b = tmp_path / "from_env", tmp_path / "from_json"
    a.mkdir()
    b.mkdir()
    return a, b


def write_json(path, obj):
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")
    return path


def test_env_wins_over_json(tmp_path, dirs):
    a, b = dirs
    js = write_json(tmp_path / "local_paths.json", {"data_dir": str(b)})
    assert data_dir(environ={ENV_VAR: str(a)}, local_paths=js) == a.resolve()


def test_blank_env_falls_to_json(tmp_path, dirs):
    _, b = dirs
    js = write_json(tmp_path / "local_paths.json", {"data_dir": str(b)})
    assert data_dir(environ={ENV_VAR: "  "}, local_paths=js) == b.resolve()


def test_json_used_when_no_env(tmp_path, dirs):
    _, b = dirs
    js = write_json(tmp_path / "local_paths.json", {"data_dir": str(b)})
    assert data_dir(environ={}, local_paths=js) == b.resolve()


def test_json_relative_path_is_relative_to_json(tmp_path, dirs):
    js = write_json(tmp_path / "local_paths.json", {"data_dir": "from_json"})
    assert data_dir(environ={}, local_paths=js) == dirs[1].resolve()


def test_json_with_bom_ok(tmp_path, dirs):
    # Notepad บน Windows ชอบใส่ BOM
    js = tmp_path / "local_paths.json"
    js.write_text(json.dumps({"data_dir": str(dirs[1])}), encoding="utf-8-sig")
    assert data_dir(environ={}, local_paths=js) == dirs[1].resolve()


def test_env_missing_dir_errors_without_fallback(tmp_path, dirs):
    _, b = dirs
    js = write_json(tmp_path / "local_paths.json", {"data_dir": str(b)})
    with pytest.raises(DataDirError, match=ENV_VAR):
        data_dir(environ={ENV_VAR: str(tmp_path / "nope")}, local_paths=js)


def test_env_missing_dir_ok_when_not_required(tmp_path):
    p = tmp_path / "later"
    assert data_dir(environ={ENV_VAR: str(p)}, local_paths=tmp_path / "x.json", must_exist=False) == p.resolve()


@pytest.mark.parametrize("content", [
    "{not json",
    {"other": "x"},
    {"data_dir": ""},
    {"data_dir": 123},
    ["C:/data"],
])
def test_bad_json_errors(tmp_path, content):
    js = write_json(tmp_path / "local_paths.json", content)
    with pytest.raises(DataDirError, match="local_paths.json"):
        data_dir(environ={}, local_paths=js)


def test_json_missing_dir_errors(tmp_path):
    js = write_json(tmp_path / "local_paths.json", {"data_dir": str(tmp_path / "nope")})
    with pytest.raises(DataDirError, match="ไม่มีอยู่"):
        data_dir(environ={}, local_paths=js)


def test_nothing_configured_errors_with_hint(tmp_path):
    with pytest.raises(DataDirError) as ei:
        data_dir(environ={}, local_paths=tmp_path / "missing.json")
    msg = str(ei.value)
    assert ENV_VAR in msg and "local_paths.json" in msg and "วิธีแก้" in msg


def test_ensure_layout_creates_only_missing(tmp_path):
    (tmp_path / "raw").mkdir()
    keep = tmp_path / "raw" / "keep.txt"
    keep.write_text("x")
    created = ensure_layout(tmp_path)
    assert {p.name for p in created} == set(SUBDIRS) - {"raw"}
    assert all((tmp_path / n).is_dir() for n in SUBDIRS)
    assert keep.read_text() == "x"
    assert ensure_layout(tmp_path) == []  # รอบสองไม่สร้างอะไร


def test_ensure_layout_file_in_the_way(tmp_path):
    (tmp_path / "zips").write_text("not a dir")
    with pytest.raises(DataDirError):
        ensure_layout(tmp_path)
