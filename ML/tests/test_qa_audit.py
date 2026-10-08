import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from roastml.decode import decode_image
from tools.qa_audit import audit_one, crop_roi, detail_metrics, find_near_duplicates, focus_region, load_trainval, safe_path
from tools import qa_audit


def row(path="raw/image.png", source="ontoum224", label="dark"):
    return {"path": path, "source": source, "label": label, "label_orig": "Dark", "split": "trainval", "group": "g1", "roi": "", "md5": "", "phash": ""}


def test_trainval_selector_never_passes_frozen_image(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row().keys())
        writer.writeheader()
        writer.writerow(row())
        writer.writerow({**row(path="frozen-do-not-read.jpg"), "split": "test"})
    rows, stats = load_trainval(manifest)
    assert [r["path"] for r in rows] == ["raw/image.png"]
    assert stats["split_counts_metadata_only"]["test"] == 1
    with pytest.raises(ValueError, match="refuses"):
        audit_one(tmp_path, {**row(), "split": "test"}, {})


def test_decode_failure_is_hard_but_dark_clip_is_review_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.png").write_bytes(b"broken")
    result, _ = audit_one(tmp_path, row("raw/bad.png"), {})
    assert result["hard_exclude"] and "decode_failed" in result["flags"]
    Image.new("RGB", (200, 200), (0, 0, 0)).save(raw / "image.png")
    result, _ = audit_one(tmp_path, row(), {})
    assert not result["hard_exclude"]
    assert "black_clipping_review" in result["flags"]
    assert result["label"] == "dark"


def test_mandatory_agtron_roi_and_exif_coordinates(tmp_path):
    path = tmp_path / "image.jpg"
    image = Image.new("RGB", (120, 80), (90, 60, 40))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    decoded = decode_image(path.read_bytes())
    assert decoded.orig_size == (80, 120)
    r = row("image.jpg", "agtron")
    with pytest.raises(ValueError, match="missing"):
        crop_roi(decoded, r)
    assert crop_roi(decoded, {**r, "roi": "0 90 70 115"}).shape[:2] == (25, 70)
    result, _ = audit_one(tmp_path, {**r, "roi": "0 0 90 110"}, {})
    assert result["hard_exclude"]


def test_blur_scores_record_resolution_and_low_texture_is_ambiguous():
    image = np.full((128, 128, 3), 80, np.uint8)
    low = detail_metrics(image, 100)
    high = detail_metrics(image, 600)
    assert low["blur_laplacian_threshold_for_resolution"] != high["blur_laplacian_threshold_for_resolution"]
    assert low["low_texture_review"] and not low["blur_review"]


def test_near_duplicate_conflicting_labels_only_flag():
    records = [{**row(label="light"), "flags": [], "reasons": [], "hard_exclude": False, "qa_view_phash": "0000000000000001", "md5_actual": "a"},
               {**row("other.png", "rf_boos", "dark"), "flags": [], "reasons": [], "hard_exclude": False, "qa_view_phash": "0000000000000001", "md5_actual": "b"}]
    pairs = find_near_duplicates(records, [np.zeros((64, 64), np.uint8)] * 2)
    assert len(pairs) == 1
    assert all("near_duplicate_label_ambiguity" in r["flags"] and not r["hard_exclude"] for r in records)


def test_manifest_path_cannot_escape_data(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../elsewhere.jpg")


def test_focus_uses_bean_size_and_not_background():
    rgb = np.full((640, 640, 3), 255, np.uint8)
    rgb[288:352, 288:352] = 75
    focus, short = focus_region(rgb, [(0, 0.5, 0.5, 0.1, 0.1)], (640, 640), 640)
    assert focus.shape[:2] == (64, 64)
    assert short == 64
    assert focus.max() == 75
    assert detail_metrics(focus, short)["low_texture_review"]


def test_invalid_bbox_and_label_mismatch_do_not_exclude(tmp_path):
    directory = tmp_path / "raw" / "rf_boos" / "train"
    (directory / "images").mkdir(parents=True)
    (directory / "labels").mkdir()
    Image.new("RGB", (100, 100), (95, 60, 20)).save(directory / "images" / "image.png")
    (directory / "labels" / "image.txt").write_text("0 0.95 0.5 0.3 0.2\n")
    r = row("raw/rf_boos/train/images/image.png", "rf_boos", "dark")
    result, _ = audit_one(tmp_path, r, {"rf_boos": ["Light Roast"]})
    assert not result["hard_exclude"] and result["label"] == "dark"
    assert {"bbox_outside_image", "label_annotation_mismatch"} <= set(result["flags"])


def test_inner_guard_is_transitive_keeps_labels_and_never_merges_sources(tmp_path, monkeypatch):
    ml = tmp_path / "ML"
    result_dir = ml / "results" / "qa_test_guard"
    result_dir.mkdir(parents=True)
    (result_dir / "policy.json").write_text("{}")
    root = tmp_path / "data"
    cache = root / "cache" / "qa_test_guard"
    cache.mkdir(parents=True)
    (root / "manifest.csv").write_text("immutable manifest reference")
    records = [{**row("a.png", label="dark"), "group": "g1"},
               {**row("b.png", label="light"), "group": "g2"},
               {**row("c.png", label="medium"), "group": "g3"},
               {**row("d.png", source="rf_boos"), "group": "g4"}]
    (cache / "ledger.json").write_text(json.dumps(records))
    with (cache / "near_duplicate_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("a", "b", "source_a", "source_b"))
        writer.writeheader()
        writer.writerows([{"a": "a.png", "b": "b.png", "source_a": "ontoum224", "source_b": "ontoum224"},
                          {"a": "b.png", "b": "c.png", "source_a": "ontoum224", "source_b": "ontoum224"},
                          {"a": "c.png", "b": "d.png", "source_a": "ontoum224", "source_b": "rf_boos"}])
    monkeypatch.setattr(qa_audit, "ML_DIR", ml)
    result = qa_audit.build_inner_groups(root, result_dir)
    mapping = result["group_to_inner_group"]
    assert mapping["g1"] == mapping["g2"] == mapping["g3"]
    assert mapping["g4"] != mapping["g1"]
    assert result["residual_within_source_candidate_cross_component_pairs"] == 0
    assert result["unmerged_cross_source_candidate_pairs"] == 1
    assert (root / "manifest.csv").read_text() == "immutable manifest reference"
    assert json.loads((cache / "ledger.json").read_text()) == records
