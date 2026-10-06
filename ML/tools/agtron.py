"""agtron (Rodrigues 2022 "Coffecolor") — mapping, split ตามมือถือ, ROI crop · ตัดสินแล้วใน results/split_decisions.md

- mapping ตายตัว (Coffee Review / M-Basic): 75→light · 65,55→medium · 45,35,25→dark — ห้ามปรับตามผลโมเดล
- split: สุ่ม 2 จาก 7 device_id ด้วย seed 20261006 → test (แช่แข็ง) · ที่เหลือ → trainval
- ROI crop บังคับ: พิกัด X1..Y2 อ้างอิงภาพ **หลัง** หมุนตาม EXIF (ตรวจด้วยตาแล้ว: ภาพ orientation 6/8 ถ้าไม่หมุน
  crop จะโดนขอบจาน/กระดาษ) · ห้ามใช้ภาพเต็ม (มีป้ายเลข Agtron + "FLA" อยู่ในภาพ)
- name_paper = ภาพกระดาษเปล่า ไม่ใช่ตัวอย่าง → เก็บเป็นคอลัมน์เสริมเท่านั้น

CLI: python -m tools.agtron --contact-sheets  → $ROAST_DATA_DIR/contact_sheets/agtron_roi/<device>.jpg + สถิติพิกเซลขาว
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from roastml.paths import CONTACT_SHEETS, RAW, data_dir

LABEL_MAP = {75: "light", 65: "medium", 55: "medium", 45: "dark", 35: "dark", 25: "dark"}
TEST_SEED = 20261006
N_TEST_DEVICES = 2
IMG_SUBDIR = Path("RAW") / "RAW"


@dataclass(frozen=True)
class AgtronPhoto:
    device_id: str
    device: str
    name_coffee: str
    name_paper: str
    agtron: int
    flash: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (พิกัดหลัง exif_transpose)


def choose_test_devices(device_ids: list[str], seed: int = TEST_SEED, k: int = N_TEST_DEVICES) -> list[str]:
    """สุ่มแบบกำหนดผลได้: random.Random(seed).sample(เรียง device_id ตามตัวเลข, k) → เรียงผลตามตัวเลข"""
    ids = sorted(set(device_ids), key=int)
    return sorted(random.Random(seed).sample(ids, k), key=int)


def read_photos(base: Path) -> tuple[list[AgtronPhoto], dict]:
    """อ่าน photos.csv + devices.csv · แถวที่ name_coffee ซ้ำ: เหมือนกันทุกช่อง → เก็บ 1 · ขัดกัน → ตัดทั้งหมด
    คืน (รายการ, รายงานสิ่งที่ตัด)"""
    with open(base / "devices.csv", newline="", encoding="utf-8-sig") as f:
        devices = {r["device_id"]: r["name"] for r in csv.DictReader(f, delimiter=";")}
    with open(base / "photos.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_name[r["name_coffee"]].append(r)
    photos, report = [], {"dup_rows_identical": 0, "dup_rows_conflicting_dropped": 0, "unknown_agtron_dropped": 0}
    for name, rs in by_name.items():
        if len(rs) > 1:
            if all(r == rs[0] for r in rs):
                report["dup_rows_identical"] += len(rs) - 1
            else:
                report["dup_rows_conflicting_dropped"] += len(rs)
                continue
        r = rs[0]
        ag = int(r["agtron"])
        if ag not in LABEL_MAP:
            report["unknown_agtron_dropped"] += 1
            continue
        box = tuple(int(r[k]) for k in ("X1", "Y1", "X2", "Y2"))
        photos.append(AgtronPhoto(r["device_id"], devices.get(r["device_id"], "?"), name, r["name_paper"],
                                  ag, r["flash"], box))
    return photos, report


def load_roi(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    """เปิดภาพ → หมุนตาม EXIF → crop ROI (RGB) · กรอบหลุดภาพ → ValueError (ไม่ crop แบบเติมขอบเงียบๆ)"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height):
        raise ValueError(f"ROI {box} อยู่นอกภาพ {im.size}: {path.name}")
    return im.crop(box)


def white_fraction(img: Image.Image) -> float:
    """สัดส่วนพิกเซลสว่างและไม่มีสี (กระดาษ/ป้าย) — ใช้คัด crop ที่อาจมีพื้นกระดาษหลุดเข้ามา"""
    a = np.asarray(img.resize((128, 128)), dtype=np.int16)
    bright = a.min(axis=2) > 170
    gray = (a.max(axis=2) - a.min(axis=2)) < 30
    return float((bright & gray).mean())


def contact_sheets(root: Path, thumb: int = 96, per_row: int = 12) -> dict:
    base = root / RAW / "agtron"
    photos, _ = read_photos(base)
    out_dir = root / CONTACT_SHEETS / "agtron_roi"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    by_dev: dict[str, list[AgtronPhoto]] = defaultdict(list)
    for p in photos:
        by_dev[p.device_id].append(p)
    for dev, ps in sorted(by_dev.items(), key=lambda kv: int(kv[0])):
        ps.sort(key=lambda p: (p.agtron, p.name_coffee))
        groups = defaultdict(list)
        for p in ps:
            groups[p.agtron].append(p)
        rows_needed = sum(-(-len(g) // per_row) for g in groups.values())
        sheet = Image.new("RGB", (per_row * (thumb + 2) + 60, rows_needed * (thumb + 2) + 4), "white")
        d = ImageDraw.Draw(sheet)
        y = 2
        wf = []
        for ag, g in sorted(groups.items()):
            d.text((2, y + 4), f"A{ag}", fill="black")
            for i, p in enumerate(g):
                crop = load_roi(base / IMG_SUBDIR / p.name_coffee, p.box)
                w = white_fraction(crop)
                wf.append((w, p.name_coffee, ag))
                t = crop.resize((thumb, thumb))
                if w > 0.02:  # ทำกรอบแดงให้เห็นตอนตรวจด้วยตา
                    ImageDraw.Draw(t).rectangle([0, 0, thumb - 1, thumb - 1], outline="red", width=3)
                sheet.paste(t, (60 + (i % per_row) * (thumb + 2), y + (i // per_row) * (thumb + 2)))
            y += -(-len(g) // per_row) * (thumb + 2)
        name = ps[0].device.replace(" ", "_")
        sheet.save(out_dir / f"dev{dev}_{name}.jpg", quality=85)
        wf.sort(reverse=True)
        stats[f"dev{dev}_{name}"] = {"n": len(ps), "white_frac_max": round(wf[0][0], 4),
                                     "n_white_frac_gt_0.02": sum(w > 0.02 for w, _, _ in wf)}
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--contact-sheets", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = args.data_dir.resolve() if args.data_dir else data_dir()
    photos, report = read_photos(root / RAW / "agtron")
    devs = sorted({(p.device_id, p.device) for p in photos}, key=lambda t: int(t[0]))
    test = choose_test_devices([d for d, _ in devs])
    print("devices:", devs)
    print("test devices (seed %d):" % TEST_SEED, [(d, n) for d, n in devs if d in test])
    print("read report:", report, "· photos:", len(photos))
    print("photos per split:", Counter("test" if p.device_id in test else "trainval" for p in photos))
    if args.contact_sheets:
        for k, v in contact_sheets(root).items():
            print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
