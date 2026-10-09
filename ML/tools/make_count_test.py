"""สร้าง count-test template ใน $ROAST_DATA_DIR/count_test/ (นอก git · ห้าม commit ภาพ/CSV)

- เลือก 25 ภาพจาก trainval ที่มีอยู่ (ไม่แตะ split=test) + จอง 15 ช่อง web (W01–W15) ให้ผู้ใช้หาเอง
- กำหนด split dev 15 / frozen 25 ระดับ group ด้วย seed **ก่อน** มีการนับ (ภาพ group เดียวกันอยู่ split เดียวกัน)
  ภาพเดิม: dev 10 / frozen 15 · web: dev 5 / frozen 10
- rf_boos มี YOLO box รายเมล็ด → เติม n_* จาก YOLO ไว้ก่อน (notes = prefilled_from_yolo ให้ผู้ใช้ตรวจ)
- agtron: ภาพต้นฉบับมีป้ายเลข Agtron → เขียน ROI crop เป็นไฟล์ใหม่ใน count_test/images/ (ไม่แตะ raw/)
- ไม่เขียนทับ: ถ้ามี count_test.csv อยู่แล้ว → หยุด (กันลบผลนับของผู้ใช้)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

from PIL import Image

from roastml.decode import decode_image
from roastml.rgb_views import crop_roi

SEED = 20261009
FIELDS = ["path", "source_url", "scene", "n_light", "n_medium", "n_dark", "n_total", "split", "notes"]
YOLO_BOOS = {0: "dark", 1: "light", 2: "medium"}  # data.yaml: ['Dark Roast', 'Light Roast', 'Medium Roast']
TARGET = {"existing": {"dev": 10, "frozen": 15}, "web": {"dev": 5, "frozen": 10}}

# แผนการเลือก (ตัดสินจาก contact sheet ก่อนนับ): (group, scene, จำนวนภาพ, หมายเหตุ)
#   rf_boos WhatsApp: 3–4 เมล็ดวางแยกบนกระดาษเทามีเงา (segment ได้ mode=full)
#   rf_boos Medium-1: มี 2 เมล็ดแตะกัน (ตัวนับ contour นับได้ 3 จาก GT 4)
#   rf_boos Mixed   : 6 เมล็ด 3 ระดับวางแยกบนพื้นขาว
#   agtron dev1–dev5: ชั้นเมล็ดเต็ม ROI (เมล็ดแตะ/ซ้อนกัน ~40–70 เมล็ดที่เห็น)
#   rf_hendi        : เมล็ดในเครื่องคั่ว มีทั้งกระจาย/แตะ/กอง
PLAN = [
    ("rf_boos:video:WhatsApp-Video-2026-05-19-at-9_14_47-PM", "flat", 2),
    ("rf_boos:video:WhatsApp-Video-2026-05-19-at-9_14_56-PM", "flat", 2),
    ("rf_boos:video:Medium-1-", "touching", 4),
    ("rf_boos:video:Mixed", "flat", 3),
    ("agtron:dev1", "pile", 2), ("agtron:dev2", "pile", 2), ("agtron:dev4", "pile", 2),
    ("agtron:dev5", "pile", 2), ("agtron:dev6", "pile", 2),
]
N_HENDI = 4  # แต่ละภาพคนละ group (prefix เวลา)


def yolo_classes(img_path: Path) -> list[int]:
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not lbl.is_file():
        raise FileNotFoundError(f"no YOLO label for {img_path}")
    return [int(line.split()[0]) for line in lbl.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 5]


def spaced(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k ภาพกระจายตลอดวิดีโอ (เฟรมติดกันเกือบซ้ำ) · จุดเริ่มสุ่มด้วย seed"""
    rows = sorted(rows, key=lambda r: r["path"])
    step = len(rows) // k
    off = rng.randrange(step)
    return [rows[off + i * step] for i in range(k)]


def assign_split(groups: list[tuple[str, int]], n_dev: int, rng: random.Random) -> dict[str, str]:
    """สุ่มลำดับ group แล้วเติม dev จนได้ n_dev พอดี (ข้าม group ที่ใส่แล้วเกิน) · ที่เหลือ frozen"""
    order = groups[:]
    rng.shuffle(order)
    out, dev = {}, 0
    for g, n in order:
        if dev + n <= n_dev:
            out[g], dev = "dev", dev + n
        else:
            out[g] = "frozen"
    if dev != n_dev:
        raise RuntimeError(f"cannot reach dev={n_dev} with group sizes {groups}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--touching-from", type=Path, required=True,
                    help="predictions_old.csv จาก audit_bean_counts (75ea419): เฟรม Medium-1 ที่ contour นับน้อยกว่า GT = มีเมล็ดแตะกัน")
    args = ap.parse_args(argv)
    with args.touching_from.open(newline="", encoding="utf-8") as f:
        touching = {r["path"] for r in csv.DictReader(f)
                    if r["source"] == "rf_boos" and r["mode"] == "beans" and r["n_beans"] != ""
                    and int(r["n_beans"]) < int(r["n_gt"])}
    root = args.data_dir.resolve()
    out_dir = root / "count_test"
    csv_path = out_dir / "count_test.csv"
    if csv_path.exists():
        raise SystemExit(f"{csv_path} exists — refusing to overwrite user counts")
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        man = [r for r in csv.DictReader(f) if r["split"] == "trainval"]
    rng = random.Random(SEED)

    picks: list[tuple[dict, str]] = []
    for group, scene, k in PLAN:
        rows = [r for r in man if r["group"] == group]
        if group == "rf_boos:video:Medium-1-":  # เฉพาะเฟรมที่มีเมล็ดแตะกัน (มี 2 blob รวมเป็น 1)
            rows = [r for r in rows if r["path"] in touching]
        if group.startswith("agtron:"):  # คนละระดับคั่วใน device เดียวกัน
            levels = sorted({r["label_orig"] for r in rows})
            chosen = rng.sample(levels, k)
            for lv in chosen:
                picks.append((rng.choice(sorted((r for r in rows if r["label_orig"] == lv), key=lambda r: r["path"])), scene))
            continue
        picks += [(r, scene) for r in spaced(rows, k, rng)]
    hendi_groups = sorted({r["group"] for r in man if r["source"] == "rf_hendi" and r["label"] in ("light", "medium", "dark")})
    for g in rng.sample(hendi_groups, N_HENDI):
        picks.append((rng.choice(sorted((r for r in man if r["group"] == g and r["label"] in ("light", "medium", "dark")),
                                        key=lambda r: r["path"])), "pile"))
    if len(picks) != 25:
        raise RuntimeError(f"expected 25 picks, got {len(picks)}")

    sizes: dict[str, int] = {}
    for r, _ in picks:
        sizes[r["group"]] = sizes.get(r["group"], 0) + 1
    split_existing = assign_split(sorted(sizes.items()), TARGET["existing"]["dev"], rng)
    web_slots = [f"W{i:02d}" for i in range(1, 16)]
    split_web = assign_split([(w, 1) for w in web_slots], TARGET["web"]["dev"], rng)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows_out, meta = [], []
    for i, (r, scene) in enumerate(picks, 1):
        p = root / r["path"]
        notes, counts = [], {"light": "", "medium": "", "dark": ""}
        path = r["path"]
        if r["source"] == "agtron":
            rgb = crop_roi(decode_image(p.read_bytes()), r["roi"])
            path = f"count_test/images/E{i:02d}_agtron_roi.jpg"
            buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=95)
            (root / path).write_bytes(buf.getvalue())
            notes.append(f"agtron ROI crop of {r['path']} (label_orig={r['label_orig']}, single roast={r['label']})")
        elif r["source"] == "rf_boos":
            cls = [YOLO_BOOS[c] for c in yolo_classes(p)]
            counts = {c: cls.count(c) for c in ("light", "medium", "dark")}
            notes.append("prefilled_from_yolo (ตรวจ/แก้ได้)")
        else:
            notes.append(f"single roast={r['label']}")
        n_tot = sum(counts.values()) if r["source"] == "rf_boos" else ""
        rows_out.append({"path": path, "source_url": r.get("url", ""), "scene": scene,
                         "n_light": counts["light"], "n_medium": counts["medium"], "n_dark": counts["dark"],
                         "n_total": n_tot, "split": split_existing[r["group"]], "notes": " · ".join(notes)})
        meta.append({"id": f"E{i:02d}", "path": path, "manifest_path": r["path"], "source": r["source"],
                     "group": r["group"], "label": r["label"], "scene": scene})
    for w in web_slots:
        rows_out.append({"path": f"count_test/images/{w}_<ชื่อไฟล์>", "source_url": "", "scene": "", "n_light": "",
                         "n_medium": "", "n_dark": "", "n_total": "", "split": split_web[w], "notes": f"slot {w} (web)"})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel อ่านภาษาไทยได้
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows_out)
    (out_dir / "count_test_meta.json").write_text(json.dumps({
        "seed": SEED, "target": TARGET, "plan": PLAN, "n_hendi": N_HENDI, "existing": meta,
        "split_existing_by_group": split_existing, "split_web_by_slot": split_web,
        "rule": "group-level split decided before counting; split=test never read"}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({"n": len(rows_out), "dev": sum(r["split"] == "dev" for r in rows_out),
                      "frozen": sum(r["split"] == "frozen" for r in rows_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
