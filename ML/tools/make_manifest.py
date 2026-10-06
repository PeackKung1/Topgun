"""สร้าง $ROAST_DATA_DIR/manifest.csv จาก raw/ ทุก source (ml-spec ข้อ 5 + results/split_decisions.md)

คอลัมน์: path,label,label_orig,source,group,split,license,url,md5,phash + เสริม: roi,device,paper_path
- label: light | medium | dark (target) · green | empty | mixed (เก็บไว้ ไม่ใช่ target)
- split: trainval (ประเมินด้วย LOSO ตาม source) · test = agtron 2 เครื่องที่สุ่มไว้ (แช่แข็ง)
- md5 = ไฟล์ต้นฉบับ · phash = ภาพที่ใช้จริง (agtron = ROI crop, อื่นๆ = ภาพเต็ม)
- dedupe รอบนี้ด้วย md5 อย่างเดียว: เก็บ 1 ไฟล์ต่อ md5 (ลำดับ source ตาม SOURCE_PRIORITY → split เดิม train>valid>test → path)
  md5 เดียวกันแต่ label ต่างกัน → ตัดทั้งกลุ่ม
- แถวที่ตัดทิ้งทั้งหมด (พร้อมเหตุผล) → $ROAST_DATA_DIR/cache/manifest_dropped.csv · ตัวเลขสรุป → ML/results/manifest_summary.json

ใช้: python -m tools.make_manifest [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from roastml.paths import CACHE, RAW, data_dir
from tools import agtron as ag
from tools.imghash import md5_file, phash_array
from tools.index_sources import Item, index_source

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
COLUMNS = ["path", "label", "label_orig", "source", "group", "split", "license", "url", "md5", "phash",
           "roi", "device", "paper_path"]
TARGETS = ("light", "medium", "dark")
NON_TARGETS = ("green", "empty", "mixed")
# ลำดับความสำคัญเวลา md5 ซ้ำข้าม source (ml-spec: ontoum224 > rf_* ตามลำดับในตาราง)
SOURCE_PRIORITY = ("ontoum224", "rf_hendi", "rf_devlong", "rf_robusta", "rf_boos", "agtron")
SPLIT_ORIG_PRIORITY = {"train": 0, "valid": 1, "test": 2}

LICENSE = {"ontoum224": "CC BY-SA 4.0", "rf_hendi": "CC BY 4.0", "rf_devlong": "CC BY 4.0",
           "rf_robusta": "CC BY 4.0", "rf_boos": "CC BY 4.0", "agtron": "Unknown"}
URL = {  # Roboflow: จาก README.dataset.txt ใน zip · Kaggle: owner/slug ตาม ml-spec (ontoum224 ยังไม่รู้ slug)
    "ontoum224": "",
    "rf_hendi": "https://universe.roboflow.com/hendi-hart-wysup/coffee-bean-s7gsl",
    "rf_devlong": "https://universe.roboflow.com/devlong-mwkgm/coffee-roast",
    "rf_robusta": "https://universe.roboflow.com/robusta/all_dataset-jflsn",
    "rf_boos": "https://universe.roboflow.com/boos-workspace/roasting-level",
    "agtron": "kaggle:joaovvrodrigues/coffee-toasted-agtron",
}

# ---------------------------------------------------------------- label mapping
FOLDER_MAP = {
    "ontoum224": {"Dark": "dark", "Medium": "medium", "Light": "light", "Green": "green"},
    "rf_hendi": {"Dark": "dark", "Medium": "medium", "Light": "light", "Raw": "green", "Empty": "empty"},
    "rf_devlong": {"Dark": "dark", "Medium": "medium", "Light": "light"},
}
YOLO_MAP = {"Dark Roast": "dark", "Medium Roast": "medium", "Light Roast": "light", "Raw": "green"}
YOLO_DROP_CLASS = "Maw"  # rf_robusta: ไม่มีนิยาม → ภาพที่มีแต่ Maw ตัดทิ้ง · Maw ปนคลาสอื่น → mixed


def map_label(source: str, label_orig: str) -> tuple[str | None, str]:
    """คืน (label, เหตุผลถ้าตัด) · label None = ตัดทิ้ง"""
    if source in FOLDER_MAP:
        lab = FOLDER_MAP[source].get(label_orig)
        return (lab, "") if lab else (None, f"unknown_class:{label_orig}")
    if label_orig == "(no_label)":
        return None, "yolo_no_label"
    classes = set(label_orig.split("|"))
    if YOLO_DROP_CLASS in classes:
        return (None, "maw_only") if classes == {YOLO_DROP_CLASS} else ("mixed", "")
    unknown = classes - set(YOLO_MAP)
    if unknown:
        return None, f"unknown_class:{'|'.join(sorted(unknown))}"
    mapped = {YOLO_MAP[c] for c in classes}
    return (mapped.pop(), "") if len(mapped) == 1 else ("mixed", "")


# ---------------------------------------------------------------- group
_HENDI_TIME = re.compile(r"^(\d{8}_\d{6})")
_VIDEO = re.compile(r"^(.*?)_mp4-\d+")
_LINE_ALBUM = re.compile(r"^(LINE_ALBUM_\d+)_")
_MSG_DATE = re.compile(r"^\d+-[0-9a-f]{32}-(\d{8})_JPG$")


def group_of(source: str, stem: str, path: str) -> str:
    """หน่วยที่ห้ามแยกข้าม split (ml-spec ข้อ 5) · ไม่เข้าเงื่อนไขไหน → ไฟล์"""
    if source == "rf_hendi" and (m := _HENDI_TIME.match(stem)):
        return f"{source}:{m.group(1)}"
    if source in ("rf_robusta", "rf_boos"):
        if m := _VIDEO.match(stem):
            return f"{source}:video:{m.group(1)}"
        if m := _LINE_ALBUM.match(stem):
            return f"{source}:{m.group(1)}"
        if m := _MSG_DATE.match(stem):
            return f"{source}:msg:{m.group(1)}"
    return f"{source}:file:{path}"


# ---------------------------------------------------------------- rows
@dataclass
class Row:
    path: str
    label: str
    label_orig: str
    source: str
    group: str
    split: str
    license: str
    url: str
    md5: str = ""
    phash: str = ""
    roi: str = ""
    device: str = ""
    paper_path: str = ""
    split_orig: str = field(default="", repr=False)  # ใช้จัดลำดับตอน dedupe ไม่เขียนลง manifest


def rows_from_items(root: Path, items: list[Item], dropped: list[dict]) -> list[Row]:
    rows = []
    for it in items:
        label, why = map_label(it.source, it.label_orig)
        if label is None:
            dropped.append({"path": it.path, "source": it.source, "label_orig": it.label_orig, "reason": why})
            continue
        rows.append(Row(it.path, label, it.label_orig, it.source, group_of(it.source, it.stem, it.path),
                        "trainval", LICENSE[it.source], URL[it.source], split_orig=it.split_orig))
    return rows


def rows_agtron(root: Path, dropped: list[dict]) -> tuple[list[Row], dict]:
    base = root / RAW / "agtron"
    photos, report = ag.read_photos(base)
    test_devs = ag.choose_test_devices([p.device_id for p in photos])
    listed = {p.name_coffee for p in photos} | {p.name_paper for p in photos}
    img_dir = base / ag.IMG_SUBDIR
    for f in sorted(img_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.name not in listed:
            dropped.append({"path": f.relative_to(root).as_posix(), "source": "agtron", "label_orig": "",
                            "reason": "not_in_photos_csv"})
    rows = []
    for p in photos:
        rows.append(Row(
            path=(img_dir / p.name_coffee).relative_to(root).as_posix(),
            label=ag.LABEL_MAP[p.agtron], label_orig=f"agtron_{p.agtron}", source="agtron",
            group=f"agtron:dev{p.device_id}", split="test" if p.device_id in test_devs else "trainval",
            license=LICENSE["agtron"], url=URL["agtron"],
            roi=" ".join(map(str, p.box)), device=f"{p.device_id}:{p.device}",
            paper_path=(img_dir / p.name_paper).relative_to(root).as_posix(),
        ))
    report["test_devices"] = [f"{p.device_id}:{p.device}" for p in photos if p.device_id in test_devs]
    report["test_devices"] = sorted(set(report["test_devices"]))
    return rows, report


def fill_hashes(root: Path, rows: list[Row]) -> None:
    for r in rows:
        p = root / r.path
        r.md5 = md5_file(p)
        if r.roi:
            img = ag.load_roi(p, tuple(int(v) for v in r.roi.split()))
            gray = np.asarray(img.convert("L"))
        else:
            from tools.imghash import load_gray
            gray = load_gray(p)
        r.phash = f"{phash_array(gray):016x}"


def dedupe_md5(rows: list[Row], dropped: list[dict]) -> tuple[list[Row], dict]:
    by_md5: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_md5[r.md5].append(r)
    keep, stats = [], Counter()
    for md5, rs in by_md5.items():
        if len(rs) == 1:
            keep.append(rs[0])
            continue
        if len({r.label for r in rs}) > 1:
            stats["groups_label_conflict_dropped"] += 1
            for r in rs:
                dropped.append({"path": r.path, "source": r.source, "label_orig": r.label_orig,
                                "reason": f"md5_label_conflict:{md5}"})
            continue
        rs.sort(key=lambda r: (SOURCE_PRIORITY.index(r.source), SPLIT_ORIG_PRIORITY.get(r.split_orig, 9), r.path))
        keep.append(rs[0])
        stats["groups_deduped"] += 1
        if len({r.source for r in rs}) > 1:
            stats["groups_cross_source"] += 1
        for r in rs[1:]:
            stats[f"removed:{r.source}"] += 1
            dropped.append({"path": r.path, "source": r.source, "label_orig": r.label_orig,
                            "reason": f"md5_dup_of:{rs[0].path}"})
    return keep, dict(stats)


def summarize(rows: list[Row], dropped: list[dict]) -> dict:
    sls = Counter((r.source, r.label, r.split) for r in rows)
    table: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for (s, l, sp), n in sorted(sls.items()):
        table[s][sp][l] = n
    return {
        "n_rows": len(rows),
        "source_x_split_x_label": {s: dict(v) for s, v in table.items()},
        "label_totals": dict(Counter(r.label for r in rows)),
        "split_totals": dict(Counter(r.split for r in rows)),
        "n_groups_per_source": {s: len({r.group for r in rows if r.source == s})
                                for s in sorted({r.source for r in rows})},
        "dropped_by_reason": dict(Counter(d["reason"].split(":")[0] for d in dropped)),
        "dropped_by_source_reason": dict(Counter(f"{d['source']}:{d['reason'].split(':')[0]}" for d in dropped)),
    }


def write_csv_atomic(path: Path, rows: list[dict], columns: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def build(root: Path, sources: tuple[str, ...] = SOURCE_PRIORITY) -> tuple[list[Row], list[dict], dict]:
    dropped: list[dict] = []
    rows: list[Row] = []
    extra = {}
    for s in sources:
        if not (root / RAW / s).is_dir():
            raise FileNotFoundError(f"ไม่มี raw/{s} — แตก zip ก่อน (tools/extract_zips.py)")
        if s == "agtron":
            r, extra["agtron"] = rows_agtron(root, dropped)
            rows += r
        else:
            rows += rows_from_items(root, index_source(root, s), dropped)
    fill_hashes(root, rows)
    rows, extra["dedupe_md5"] = dedupe_md5(rows, dropped)
    rows.sort(key=lambda r: (SOURCE_PRIORITY.index(r.source), r.path))
    return rows, dropped, extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = args.data_dir.resolve() if args.data_dir else data_dir()

    rows, dropped, extra = build(root)
    write_csv_atomic(root / "manifest.csv", [asdict(r) for r in rows], COLUMNS)
    (root / CACHE).mkdir(exist_ok=True)
    write_csv_atomic(root / CACHE / "manifest_dropped.csv", dropped, ["path", "source", "label_orig", "reason"])
    summary = summarize(rows, dropped) | extra
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n→ {root / 'manifest.csv'} ({len(rows)} แถว) · dropped {len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
