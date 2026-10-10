"""สร้าง manifest.csv จาก 4 source ที่มี (ไม่มี rf_hendi) แล้วเทียบกับ results/manifest_summary.json ที่อยู่ใน git

    python results/yolo_bean_retrain_20261010/scripts/build_manifest_4src.py        (cd ML, ตั้ง ROAST_DATA_DIR)

ไม่ใช่ manifest ต้นฉบับ: เครื่องนี้ไม่ได้รับ manifest.csv / rf_hendi.zip จึงเรียก tools.make_manifest.build() โดยตัด rf_hendi ออก
- ไม่เรียก make_manifest.main() เพราะจะเขียนทับ results/manifest_summary.json (ไฟล์อ้างอิงใน git)
- เขียน manifest.csv เฉพาะเมื่อจำนวนแถวต่อ source/split/label, จำนวน group และเครื่อง test ของ agtron ตรงกับ summary เดิมทุกตัว
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML))

from roastml.paths import data_dir  # noqa: E402
from tools.make_manifest import COLUMNS, build, summarize, write_csv_atomic  # noqa: E402

SOURCES = ("ontoum224", "rf_robusta", "rf_boos", "agtron")


def main() -> int:
    root = data_dir()
    dest = root / "manifest.csv"
    if dest.exists():
        raise SystemExit(f"{dest} มีอยู่แล้ว — ไม่เขียนทับ")
    ref = json.loads((ML / "results" / "manifest_summary.json").read_text(encoding="utf-8"))
    rows, dropped, extra = build(root, SOURCES)
    got = summarize(rows, dropped) | extra
    diffs = []
    for s in SOURCES:
        if got["source_x_split_x_label"].get(s) != ref["source_x_split_x_label"][s]:
            diffs.append({"source": s, "got": got["source_x_split_x_label"].get(s), "ref": ref["source_x_split_x_label"][s]})
        if got["n_groups_per_source"].get(s) != ref["n_groups_per_source"][s]:
            diffs.append({"source": s, "groups_got": got["n_groups_per_source"].get(s), "groups_ref": ref["n_groups_per_source"][s]})
    if got["agtron"]["test_devices"] != ref["agtron"]["test_devices"]:
        diffs.append({"agtron_test_devices": got["agtron"]["test_devices"], "ref": ref["agtron"]["test_devices"]})
    report = {"sources": SOURCES, "n_rows": got["n_rows"], "n_rows_ref_without_rf_hendi": ref["n_rows"] - sum(
        n for sp in ref["source_x_split_x_label"]["rf_hendi"].values() for n in sp.values()),
        "source_x_split_x_label": got["source_x_split_x_label"], "n_groups_per_source": got["n_groups_per_source"],
        "agtron": got["agtron"], "dedupe_md5": got["dedupe_md5"], "dropped_by_source_reason": got["dropped_by_source_reason"],
        "cross_source_near_dups": got["cross_source_near_dups"], "diffs_vs_tracked_summary": diffs}
    out = Path(__file__).resolve().parents[1] / "manifest_4src_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if diffs:
        print("FAIL: ไม่ตรงกับ manifest_summary.json — ไม่เขียน manifest.csv", file=sys.stderr)
        return 3
    write_csv_atomic(dest, [asdict(r) for r in rows], COLUMNS)
    print(f"-> {dest} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
