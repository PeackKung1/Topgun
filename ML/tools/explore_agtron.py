"""สำรวจโครงสร้าง raw/agtron (Gate Agtron, ml-spec ข้อ 5) — นับอย่างเดียว ไม่รันโมเดล ไม่ตั้ง split ไม่ map label

ข้อมูล: photos.csv (device_id;name_coffee;name_paper;agtron;flash;X1;Y1;X2;Y2;H) · devices.csv · RAW/RAW/*.jpg
ผล: ตัวเลขล้วน → ML/results/agtron_explore.json (license Unknown → ห้ามมีรูปหรือชื่อไฟล์ใน repo)

ใช้: python tools/explore_agtron.py [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from roastml.paths import RAW, data_dir

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _nested(counter: dict) -> dict:
    return {str(k): dict(sorted(v.items())) for k, v in sorted(counter.items())}


def explore(base: Path) -> dict:
    devices = {d["device_id"]: d["name"] for d in read_csv(base / "devices.csv")}
    photos = read_csv(base / "photos.csv")
    img_dir = base / "RAW" / "RAW"
    files = {p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}

    coffee = [r["name_coffee"] for r in photos]
    papers = {r["name_paper"] for r in photos}
    dev_name = lambda r: devices.get(r["device_id"], f"?{r['device_id']}")  # noqa: E731

    # ภาพกาแฟต่อค่า Agtron / มือถือ / flash
    by_agtron = Counter(int(r["agtron"]) for r in photos)
    dev_x_agtron: dict[str, Counter] = defaultdict(Counter)
    flash_x_agtron: dict[str, Counter] = defaultdict(Counter)
    for r in photos:
        dev_x_agtron[dev_name(r)][int(r["agtron"])] += 1
        flash_x_agtron[r["flash"]][int(r["agtron"])] += 1

    # "sample" ไม่มีคอลัมน์ตรงๆ — ผู้สมัครเป็นหน่วยกลุ่ม: name_paper (ภาพกระดาษอ้างอิงที่ใช้ร่วมกันหลายภาพ)
    per_paper: dict[str, list[dict]] = defaultdict(list)
    for r in photos:
        per_paper[r["name_paper"]].append(r)
    paper_n = Counter(len(v) for v in per_paper.values())
    paper_multi_agtron = sum(len({r["agtron"] for r in v}) > 1 for v in per_paper.values())
    paper_multi_device = sum(len({r["device_id"] for r in v}) > 1 for v in per_paper.values())
    roi_per_paper = Counter(len({(r["X1"], r["Y1"], r["X2"], r["Y2"]) for r in v}) for v in per_paper.values())
    # นับคู่ (paper, agtron) ที่ไม่ซ้ำ — paper เดียวอาจใช้กับหลายค่า Agtron
    papers_per_dev_agtron = Counter({(dev_name(r), int(r["agtron"]), r["name_paper"]) for r in photos})
    papers_per_dev_agtron = Counter((d, a) for (d, a, _p) in papers_per_dev_agtron)

    # ขนาดภาพ (อ่าน header เท่านั้น)
    sizes_by_dev: dict[str, Counter] = defaultdict(Counter)
    for r in photos:
        p = img_dir / r["name_coffee"]
        if p.is_file():
            with Image.open(p) as im:
                sizes_by_dev[dev_name(r)][f"{im.width}x{im.height}"] += 1

    roi_w = [int(r["X2"]) - int(r["X1"]) for r in photos]
    roi_h = [int(r["Y2"]) - int(r["Y1"]) for r in photos]
    return {
        "n_image_files": len(files),
        "n_rows_photos_csv": len(photos),
        "n_unique_coffee_in_csv": len(set(coffee)),
        "coffee_names_repeated_in_csv": sum(c > 1 for c in Counter(coffee).values()),
        "n_unique_paper_in_csv": len(papers),
        "coffee_in_csv_and_on_disk": len(set(coffee) & files),
        "coffee_in_csv_missing_on_disk": len(set(coffee) - files),
        "paper_on_disk": len(papers & files),
        "paper_also_listed_as_coffee": len(papers & set(coffee)),
        "files_not_in_csv": len(files - set(coffee) - papers),
        "devices": devices,
        "agtron_values": dict(sorted(by_agtron.items())),
        "coffee_images_device_x_agtron": _nested(dev_x_agtron),
        "coffee_images_flash_x_agtron": _nested(flash_x_agtron),
        "paper_groups": {
            "n_groups": len(per_paper),
            "images_per_group_hist": dict(sorted(paper_n.items())),
            "groups_with_more_than_one_agtron": paper_multi_agtron,
            "groups_with_more_than_one_device": paper_multi_device,
            "distinct_roi_per_group_hist": dict(sorted(roi_per_paper.items())),
            "groups_per_device_x_agtron": _nested(
                {d: Counter({a: c for (dd, a), c in papers_per_dev_agtron.items() if dd == d})
                 for d in {d for d, _ in papers_per_dev_agtron}}),
        },
        "image_sizes_by_device": _nested(sizes_by_dev),
        "roi_box_px": {"w_min": min(roi_w), "w_max": max(roi_w), "h_min": min(roi_h), "h_max": max(roi_h)},
        "H_column_distinct": len({r["H"] for r in photos}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    rep = explore(root / RAW / "agtron")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "agtron_explore.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
