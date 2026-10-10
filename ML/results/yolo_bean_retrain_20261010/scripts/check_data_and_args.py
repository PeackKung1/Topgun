"""ขั้น A–C: นับไฟล์ภาพที่มีจริงต่อ source, เขียน config ตัวนับที่สร้างขึ้นใหม่, คำนวณ argument ของ make_synth_beans

    python results/yolo_bean_retrain_20261010/scripts/check_data_and_args.py        (cd ML, ตั้ง ROAST_DATA_DIR)

อ่าน metadata + เช็กว่าไฟล์มีอยู่เท่านั้น ไม่เปิดภาพ
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ML = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ML))

from roastml.counter import CountConfig  # noqa: E402
from roastml.paths import data_dir  # noqa: E402
from tools.perbean_protocol import DevData  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
TARGET = {"holdout_train": 1245, "final_train": 2036, "val": 100}
VIDEO_CAP = 120


def main() -> int:
    data = DevData(data_dir())
    total, have = Counter(), Counter()
    for r in data.rows:
        total[r["source"]] += 1
        have[r["source"]] += (data.root / r["path"]).is_file()
    files = {s: {"rows": total[s], "files_present": have[s], "missing": total[s] - have[s],
                 "pct": round(100 * have[s] / total[s], 2)} for s in sorted(total)}
    negatives = len(data.empty_split()[0])
    boos_groups = Counter(r["group"] for r in data.rows if r["source"] == "rf_boos")
    real_boos = sum(min(VIDEO_CAP, n) for n in boos_groups.values())
    args = {"n_ontoum": TARGET["holdout_train"] - negatives,
            "n_boos": TARGET["final_train"] - TARGET["holdout_train"] - real_boos, "n_val": TARGET["val"]}
    cfg_path = OUT / "selected_config_reconstructed.json"
    cfg = CountConfig(metal_chroma_min=16.0, ws_separation=2.0).to_dict()
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    report = {"dev_rows": len(data.rows), "frozen_groups": sorted(data.frozen), "files_by_source": files,
              "rf_hendi_rows": total["rf_hendi"], "negatives": negatives, "rf_boos_dev_groups": dict(sorted(boos_groups.items())),
              "real_boos": real_boos, "target": TARGET, "make_synth_beans_args": args, "count_config": str(cfg_path.relative_to(ML))}
    (OUT / "data_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
