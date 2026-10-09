"""eval_beans.build เขียนได้เฉพาะ candidate ใหม่/ว่าง หรือ b1_linear_beans เดิม — ทุกอย่างอยู่ใน tmp_path"""
from __future__ import annotations

import json

import pytest

from test_linear_backend import write_model
from tools import eval_beans as eb


@pytest.fixture
def current(tmp_path, monkeypatch):
    cur = write_model(tmp_path / "models" / "current", "b1_linear")
    monkeypatch.setattr(eb, "CURRENT", cur)
    return cur


def snapshot(d):
    return {p.name: (p.read_bytes() if p.is_file() else "dir") for p in d.iterdir()}


def test_build_new_dir_then_rerun_over_own_candidate(current, tmp_path):
    out = tmp_path / "models" / "bean_candidate"
    card = eb.build(out, "run1")
    assert card["backend"] == "b1_linear_beans" and (out / "model.json").is_file()
    assert eb.build(out, "run2")["bean_run"] == "run2"  # rerun ทับ candidate ตัวเองได้


def test_build_into_empty_dir(current, tmp_path):
    out = tmp_path / "empty"; out.mkdir()
    assert eb.build(out, "run")["backend"] == "b1_linear_beans"


@pytest.mark.parametrize("target", ["current", "reference", "parent", "junk", "file"])
def test_build_refuses_other_dirs_without_touching_them(current, tmp_path, target):
    models = current.parent
    if target == "current":
        out = current
    elif target == "reference":
        out = write_model(models / "reference_b1_r2", "b1_linear")
    elif target == "parent":
        out = models  # ไม่ว่าง และไม่มี model_card.json
    elif target == "junk":
        out = models / "junk"; out.mkdir(); (out / "model_card.json").write_text("{not json")
    else:
        out = models / "a_file"; out.write_text("x")
    before = out.read_bytes() if out.is_file() else snapshot(out)
    cur_before = snapshot(current)
    with pytest.raises(SystemExit, match="refusing"):
        eb.build(out, "run")
    assert (out.read_bytes() if out.is_file() else snapshot(out)) == before
    assert snapshot(current) == cur_before
    if target == "reference":
        assert json.loads((out / "model_card.json").read_text())["backend"] == "b1_linear"
