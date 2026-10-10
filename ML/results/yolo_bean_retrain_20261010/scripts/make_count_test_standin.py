"""เขียน count_test/ แบบ stand-in ให้ DevData ใช้ได้ โดยคง frozen group ชุดเดิมของ run final_v2

    python results/yolo_bean_retrain_20261010/scripts/make_count_test_standin.py    (cd ML, ตั้ง ROAST_DATA_DIR)

ไม่ใช่ count_test ต้นฉบับ: เครื่องนี้ไม่ได้รับ count_test_meta.json / count_test.csv และ tools.make_count_test รันซ้ำไม่ได้
(ต้องมี rf_hendi + predictions_old.csv) · สิ่งเดียวที่ DevData.rows ใช้จากไฟล์นี้คือรายชื่อ group ที่ frozen
จึงคัดลอก "frozen_groups_excluded" จาก results/yolo_bean_20261010/final_v2/declaration.json มาตรงๆ
- existing = [] และ count_test.csv มีแต่หัวตาราง → ไม่มีภาพ count dev/frozen ให้ประเมิน (ใช้ได้แค่เทรน)
- ไม่เขียนทับไฟล์ที่มีอยู่
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML))

from roastml.paths import data_dir  # noqa: E402
from tools.make_count_test import FIELDS  # noqa: E402
from tools.perbean_protocol import read_csv  # noqa: E402

REF = ML / "results" / "yolo_bean_20261010" / "final_v2" / "declaration.json"


def main() -> int:
    root = data_dir()
    out = root / "count_test"
    meta_path, csv_path = out / "count_test_meta.json", out / "count_test.csv"
    if meta_path.exists() or csv_path.exists():
        raise SystemExit(f"{out} มีไฟล์อยู่แล้ว — ไม่เขียนทับ")
    frozen = json.loads(REF.read_text(encoding="utf-8"))["frozen_groups_excluded"]
    groups = {r["group"] for r in read_csv(root / "manifest.csv")}
    present = {g: g in groups for g in frozen}
    missing = [g for g, ok in present.items() if not ok and not g.startswith("rf_hendi:")]
    if missing:
        raise SystemExit(f"frozen group ไม่อยู่ใน manifest (ชื่อ group ไม่ตรงกับ run เดิม): {missing}")
    out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "standin": "reconstructed from frozen_groups_excluded of results/yolo_bean_20261010/final_v2/declaration.json; "
                   "not the original count_test_meta.json; training only",
        "existing": [], "split_existing_by_group": {g: "frozen" for g in frozen}, "split_web_by_slot": {},
        "rule": "group-level frozen split copied from the original declaration; split=test never read"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    print(json.dumps({"frozen_groups": present, "meta": str(meta_path), "csv": str(csv_path)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
