"""สร้างดัชนีภาพของ source แบบ "จัดโฟลเดอร์ตามคลาส" (ontoum224, rf_hendi, rf_devlong) และ YOLO (rf_robusta, rf_boos)
แล้วนับจำนวนภาพ / คลาส / ขนาดภาพต่อ source → ML/results/source_stats.json (ตัวเลขล้วน ไม่มีรูป)

label_orig = ชื่อคลาสเดิมของ dataset (ยังไม่ map) · YOLO: รวมคลาสของทุก bbox เป็น "A|B" (เรียงตามตัวอักษร)
split_orig = split เดิมของ dataset — เก็บไว้ตรวจซ้ำเท่านั้น เราไม่ใช้ (ประเมินด้วย LOSO)

ใช้: python tools/index_sources.py [--data-dir PATH] [--sources rf_hendi ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from roastml.paths import RAW, data_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOLDER_SOURCES = ("ontoum224", "rf_hendi", "rf_devlong")
YOLO_SOURCES = ("rf_robusta", "rf_boos")
RESULTS = Path(__file__).resolve().parent.parent / "results"

_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


@dataclass
class Item:
    source: str
    path: str  # สัมพัทธ์กับ data dir (ใช้ / เสมอ)
    split_orig: str
    label_orig: str
    stem: str  # ชื่อก่อน ".rf.<hash>" (Roboflow) หรือชื่อไฟล์ไม่รวมนามสกุล
    width: int
    height: int
    n_boxes: int | None = None  # YOLO เท่านั้น


def rf_stem(filename: str) -> str:
    """'abc_jpg.rf.0f3e.jpg' → 'abc_jpg' · ไม่ใช่ชื่อแบบ Roboflow → ชื่อไม่รวมนามสกุล"""
    base = Path(filename).stem
    return _RF_SUFFIX.sub("", base)


def _size(p: Path) -> tuple[int, int]:
    try:
        with Image.open(p) as im:
            return im.size
    except OSError:
        return (-1, -1)  # เปิดไม่ได้ → รายงานแยก


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def index_folder_source(root: Path, source: str) -> list[Item]:
    """โครง <split>/<class>/<file>"""
    base = root / RAW / source
    items = []
    for split_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for img in _images(cls_dir):
                w, h = _size(img)
                items.append(Item(source, img.relative_to(root).as_posix(), split_dir.name,
                                  cls_dir.name, rf_stem(img.name), w, h))
    return items


def read_yolo_names(data_yaml: Path) -> list[str]:
    """อ่านบรรทัด `names: [...]` จาก data.yaml (ไม่พึ่ง pyyaml)"""
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names = ast.literal_eval(line.split(":", 1)[1].strip())
            if isinstance(names, dict):  # รูปแบบ {0: 'a', 1: 'b'}
                names = [names[k] for k in sorted(names)]
            return [str(n) for n in names]
    raise ValueError(f"ไม่พบ names ใน {data_yaml}")


def index_yolo_source(root: Path, source: str) -> list[Item]:
    """โครง <split>/images/<file> + <split>/labels/<stem>.txt"""
    base = root / RAW / source
    names = read_yolo_names(base / "data.yaml")
    items = []
    for split_dir in sorted(p for p in base.iterdir() if (p / "images").is_dir()):
        for img in _images(split_dir / "images"):
            lbl = split_dir / "labels" / (img.stem + ".txt")
            classes: set[str] = set()
            n = 0
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        classes.add(names[int(line.split()[0])])
                        n += 1
            label = "|".join(sorted(classes)) if classes else "(no_label)"
            w, h = _size(img)
            items.append(Item(source, img.relative_to(root).as_posix(), split_dir.name,
                              label, rf_stem(img.name), w, h, n_boxes=n))
    return items


def index_source(root: Path, source: str) -> list[Item]:
    if source in FOLDER_SOURCES:
        return index_folder_source(root, source)
    if source in YOLO_SOURCES:
        return index_yolo_source(root, source)
    raise ValueError(f"ไม่รองรับ source {source!r}")


def stats(items: list[Item]) -> dict:
    by_label = Counter(i.label_orig for i in items)
    by_split_label: dict[str, Counter] = defaultdict(Counter)
    for i in items:
        by_split_label[i.split_orig][i.label_orig] += 1
    sizes = Counter(f"{i.width}x{i.height}" for i in items)
    out = {
        "n_images": len(items),
        "labels_orig": dict(sorted(by_label.items())),
        "split_orig_x_label": {s: dict(sorted(c.items())) for s, c in sorted(by_split_label.items())},
        "sizes_top": dict(sizes.most_common(5)),
        "n_distinct_sizes": len(sizes),
        "n_unreadable": sizes.get("-1x-1", 0),
        "n_distinct_stems": len({i.stem for i in items}),
    }
    if items and items[0].n_boxes is not None:
        out["n_classes_per_image"] = dict(sorted(Counter(i.label_orig.count("|") + 1 if i.label_orig != "(no_label)" else 0
                                                         for i in items).items()))
        out["boxes_total"] = sum(i.n_boxes or 0 for i in items)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--sources", nargs="+", default=list(FOLDER_SOURCES + YOLO_SOURCES))
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    report = {}
    for s in args.sources:
        if not (root / RAW / s).is_dir():
            print(f"[SKIP] {s}: ยังไม่มี raw/{s}")
            continue
        report[s] = st = stats(index_source(root, s))
        print(f"\n=== {s}: {st['n_images']} ภาพ · stem ไม่ซ้ำ {st['n_distinct_stems']} · ขนาด {st['sizes_top']}")
        print("  label_orig:", st["labels_orig"])
        for sp, c in st["split_orig_x_label"].items():
            print(f"  {sp}: {c}")
        if "n_classes_per_image" in st:
            print("  จำนวนคลาสต่อภาพ:", st["n_classes_per_image"], "· bbox รวม", st["boxes_total"])

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "source_stats.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
