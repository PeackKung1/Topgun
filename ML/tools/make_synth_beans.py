"""ชุดเทรน detector เมล็ดคลาสเดียวแบบสังเคราะห์ (copy-paste เมล็ดจริงลงพื้นหลัง) — เรียบ / แตะกัน / กองซ้อน

    python -m tools.make_synth_beans [--name yolo_synth_v1] [--n-ontoum 1500] [--n-boos 500] [--n-val 150]

ทำไมต้องสังเคราะห์: กรอบของ rf_robusta เป็นกรอบ "ทั้งกอง" ไม่ใช่รายเมล็ด · กรอบรายเมล็ดจริงมีแค่ rf_boos (2–6 เมล็ด พื้นขาว)
- sprite = เมล็ดจริงที่ตัดด้วย roastml.counter จาก ontoum224 (1 เมล็ด/ภาพ) และ rf_boos · เฉพาะ dev ที่ปลอดภัย (DevData)
- label = กรอบของส่วนที่มองเห็น · เมล็ดที่เห็น < 50% ไม่มี label (ตรงนิยาม "นับเมล็ดที่เห็น ≥ ครึ่งเมล็ด")
- ภาพลบจริง = rf_hendi Empty ครึ่ง tune · เฟรมจริงของ rf_boos (กรอบ YOLO เดิม) อยู่เฉพาะใน exp "final"
exp:  holdout_boos = สังเคราะห์จาก sprite ของ ontoum เท่านั้น → ทดสอบบน rf_boos จริงได้ (ไม่เคยเห็น boos เลย)
      final        = sprite ทั้งสอง source + เฟรมจริงของ rf_boos
ผลอยู่นอก git ที่ $ROAST_DATA_DIR/cache/<name>/ · ตัวเลขบนภาพสังเคราะห์ไม่ใช่หลักฐานผ่านเกณฑ์
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from roastml.counter import CountConfig, count_beans
from roastml.decode import decode_image
from roastml.paths import data_dir
from roastml.segment import SegConfig
from tools.index_sources import read_yolo_boxes
from tools.perbean_protocol import SEED, DevData

SIZE = 640
ROASTS = ("light", "medium", "dark", "green")


def extract_sprites(data: DevData, rows: list[dict], cfg: CountConfig, max_per_image: int) -> list[dict]:
    """ตัดเมล็ดจากภาพจริงด้วยตัวนับ → RGBA (alpha ขอบนุ่ม) · ข้ามภาพที่ตัวนับได้จำนวนผิดปกติ"""
    seg, sprites = SegConfig(), []
    for r in rows:
        data.assert_safe(r)
        rgb = np.asarray(decode_image((data.root / r["path"]).read_bytes()).image)
        res = count_beans(rgb, cfg, seg)
        if not 1 <= res.n <= max_per_image:
            continue
        labels = cv2.resize(res.labels.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        for i in range(1, res.n + 1):
            m = (labels == i).astype(np.uint8)
            ys, xs = np.nonzero(m)
            if len(xs) < 150 or xs.min() <= 1 or ys.min() <= 1 or xs.max() >= rgb.shape[1] - 2 or ys.max() >= rgb.shape[0] - 2:
                continue  # เล็กเกิน/ติดขอบภาพ = เมล็ดไม่ครบ
            m = cv2.erode(m, np.ones((3, 3), np.uint8))  # ตัดขอบที่ปนพื้นหลัง/เงา
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            alpha = cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.8)
            sprites.append({"rgb": rgb[y0:y1, x0:x1].copy(), "alpha": alpha, "roast": r["label"], "source": r["source"]})
    return sprites


def background(rng: np.random.Generator, real_bgs: list[np.ndarray]) -> np.ndarray:
    kind = rng.choice(["light", "light", "paper", "color", "dark", "wood", "check", "real"])
    if kind == "real" and real_bgs:
        bg = real_bgs[rng.integers(len(real_bgs))]
        return cv2.resize(bg, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    if kind in ("light", "paper"):
        base = np.array([rng.uniform(200, 250)] * 3) + rng.uniform(-10, 10, 3)
    elif kind == "dark":
        base = np.array([rng.uniform(15, 70)] * 3) + rng.uniform(-8, 8, 3)
    else:
        base = rng.uniform(70, 230, 3)
    bg = np.ones((SIZE, SIZE, 3), np.float32) * base.astype(np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    if kind == "wood":
        ang, period = rng.uniform(0, np.pi), rng.uniform(8, 40)
        bg *= (1 + 0.08 * np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / period * 2 * np.pi))[..., None]
    if kind == "check":
        cell = int(rng.integers(30, 90))
        bg *= np.where(((xx // cell + yy // cell) % 2) > 0, 1.0, rng.uniform(0.5, 0.85))[..., None]
    bg += rng.normal(0, rng.uniform(0.5, 4), bg.shape).astype(np.float32)
    return bg


def place(sprite: dict, length: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """ย่อ/ขยายให้ด้านยาว ≈ length แล้วหมุนสุ่ม → (rgb float32, alpha float32)"""
    rgb, alpha = sprite["rgb"], sprite["alpha"]
    s = length / max(rgb.shape[:2])
    w, h = max(3, round(rgb.shape[1] * s)), max(3, round(rgb.shape[0] * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    rgb, alpha = cv2.resize(rgb, (w, h), interpolation=interp), cv2.resize(alpha, (w, h), interpolation=interp)
    if rng.random() < 0.5:
        rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
    ang = rng.uniform(0, 360)
    d = int(np.ceil(np.hypot(w, h))) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (d - w) / 2, (d - h) / 2
    rgb = cv2.warpAffine(np.ascontiguousarray(rgb), M, (d, d), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(np.ascontiguousarray(alpha), M, (d, d), flags=cv2.INTER_LINEAR, borderValue=0)
    return rgb.astype(np.float32), alpha


def render(rng: np.random.Generator, sprites: list[dict], real_bgs: list[np.ndarray]) -> tuple[np.ndarray, list, dict]:
    img = background(rng, real_bgs)
    ids = np.zeros((SIZE, SIZE), np.int32)
    scene = rng.choice(["empty", "flat", "flat", "flat", "touching", "touching", "pile", "pile", "pile"])
    region = np.ones((SIZE, SIZE), bool)
    if scene == "pile":
        length = rng.uniform(20, 70)
        if rng.random() < 0.6:  # กองบนจาน/กลางภาพ
            cx, cy, rad = rng.uniform(220, 420), rng.uniform(220, 420), rng.uniform(120, 300)
            yy, xx = np.mgrid[0:SIZE, 0:SIZE]
            region = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
            if rng.random() < 0.6:
                plate = (xx - cx) ** 2 + (yy - cy) ** 2 <= (rad * rng.uniform(1.1, 1.35)) ** 2
                img[plate] = img[plate] * 0.2 + np.float32(rng.uniform(190, 250)) * 0.8
        density = rng.uniform(1.6, 3.0)  # กองจริงแทบไม่เห็นพื้น
        length = max(length, float(np.sqrt(region.sum() * density / (0.51 * 850))))  # ไม่ให้เพดานจำนวนทำให้กองโปร่ง
        n = int(region.sum() / (0.51 * length ** 2) * density)
    elif scene == "touching":
        length = rng.uniform(28, 110)
        n = int(SIZE * SIZE / (0.51 * length ** 2) * rng.uniform(0.35, 0.8))
    elif scene == "flat":
        if rng.random() < 0.3:   # ถ่ายใกล้: 1–4 เมล็ดใหญ่เกือบเต็มภาพ (v1 ไม่มี → เมล็ดเดี่ยวถูกแบ่งเป็น 2 กรอบ)
            length = rng.uniform(240, 600)
            n = int(rng.integers(1, 5)) if length < 380 else int(rng.integers(1, 3))
        else:
            length = rng.uniform(30, 240)
            n = int(np.clip(SIZE * SIZE / (0.51 * length ** 2) * rng.uniform(0.04, 0.3), 1, 60))
    else:
        length, n = 0.0, 0
    n = min(n, 900)
    roasts = rng.choice(ROASTS[:3], size=int(rng.choice([1, 1, 2, 3])), replace=False)
    pool = [s for s in sprites if s["roast"] in roasts] or sprites
    ry, rx = np.nonzero(region)
    full_area, bean_roast = [0], [""]
    for _ in range(n):
        sp = pool[rng.integers(len(pool))]
        rgb, alpha = place(sp, length * rng.uniform(0.85, 1.15), rng)
        d = alpha.shape[0]
        for _try in range(20 if scene == "flat" else 1):
            k = rng.integers(len(rx))
            x0, y0 = int(rx[k] - d // 2), int(ry[k] - d // 2)
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + d, SIZE), min(y0 + d, SIZE)
            if xb - xa < 3 or yb - ya < 3:
                continue
            a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
            if scene != "flat" or not np.any((ids[ya:yb, xa:xb] > 0) & (a > 0.3)):
                break
        else:
            continue
        if xb - xa < 3 or yb - ya < 3:
            continue
        if scene != "pile" and rng.random() < 0.7:  # เงาใต้เมล็ด
            sh = cv2.GaussianBlur(alpha, (0, 0), max(1.0, d * 0.05))
            off = max(1, int(d * 0.06))
            ys, xs = min(ya + off, SIZE - 1), min(xa + off, SIZE - 1)
            part = sh[ya - y0:ya - y0 + (min(yb + off, SIZE) - ys), xa - x0:xa - x0 + (min(xb + off, SIZE) - xs)]
            img[ys:ys + part.shape[0], xs:xs + part.shape[1]] *= (1 - 0.45 * part)[..., None]
        gain = rng.uniform(0.9, 1.1)
        img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a[..., None]) + rgb[ya - y0:yb - y0, xa - x0:xb - x0] * gain * a[..., None]
        bid = len(full_area)
        solid = a > 0.5
        ids[ya:yb, xa:xb][solid] = bid
        full_area.append(int((alpha > 0.5).sum()))
        bean_roast.append(sp["roast"])
    # แสง/WB/เบลอ ทั้งภาพ
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    ang = rng.uniform(0, 2 * np.pi)
    img *= (rng.uniform(0.7, 1.15) * (1 + rng.uniform(0, 0.25) * ((xx - 0.5) * np.cos(ang) + (yy - 0.5) * np.sin(ang))))[..., None]
    img *= rng.uniform(0.92, 1.08, 3).astype(np.float32)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    visible = np.bincount(ids.ravel(), minlength=len(full_area))
    boxes, counts = [], {r: 0 for r in ROASTS}
    nb = len(full_area)
    ys, xs = np.nonzero(ids)
    v = ids[ys, xs]
    x0 = np.full(nb, SIZE); y0 = np.full(nb, SIZE); x1 = np.full(nb, -1); y1 = np.full(nb, -1)
    np.minimum.at(x0, v, xs); np.minimum.at(y0, v, ys); np.maximum.at(x1, v, xs); np.maximum.at(y1, v, ys)
    for bid in range(1, nb):
        if full_area[bid] < 20 or visible[bid] < 0.5 * full_area[bid]:
            continue
        bx0, bx1, by0, by1 = x0[bid], x1[bid] + 1, y0[bid], y1[bid] + 1
        boxes.append(((bx0 + bx1) / 2 / SIZE, (by0 + by1) / 2 / SIZE, (bx1 - bx0) / SIZE, (by1 - by0) / SIZE))
        counts[bean_roast[bid]] += 1
    return img, boxes, {"scene": str(scene), "pasted": nb - 1, "visible_ge_half": len(boxes), "roast_counts": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="yolo_synth_v1")
    ap.add_argument("--count-config", type=Path, required=True, help="selected_config.json ของตัวนับ (ใช้ตัด sprite)")
    ap.add_argument("--n-ontoum", type=int, default=1500)
    ap.add_argument("--n-boos", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--video-cap", type=int, default=120)
    args = ap.parse_args(argv)
    data = DevData(data_dir())
    root = data.root / "cache" / args.name
    if root.exists():
        raise FileExistsError(f"มีอยู่แล้ว: {root} — ใช้ --name ใหม่")
    cfg = CountConfig.from_dict(json.loads(args.count_config.read_text(encoding="utf-8")))
    pyrng = random.Random(SEED)
    safe = sorted(data.rows, key=lambda r: r["path"])
    on_rows = [r for r in safe if r["source"] == "ontoum224" and r["label"] in ROASTS]
    boos_real = []
    for group in sorted({r["group"] for r in safe if r["source"] == "rf_boos"}):
        rows = [r for r in safe if r["group"] == group]
        boos_real += sorted(pyrng.sample(rows, min(args.video_cap, len(rows))), key=lambda r: r["path"])
    boos_sprite_rows = [r for r in boos_real if r["label"] in ROASTS][::4]
    sp_on = extract_sprites(data, on_rows, cfg, 1)
    sp_boos = extract_sprites(data, boos_sprite_rows, cfg, 6)
    negatives, _ = data.empty_split()
    negatives = sorted(negatives, key=lambda r: r["path"])
    real_bgs = [np.asarray(decode_image((data.root / r["path"]).read_bytes()).image) for r in negatives[::8]]
    for sub in ("images", "labels", "lists"):
        (root / sub).mkdir(parents=True)
    meta, lists = {}, {"synth_ontoum": [], "synth_boos": [], "synth_val": [], "real_boos": [], "negatives": []}

    def write(key: str, tag: str, img_rgb: np.ndarray, boxes, info: dict, quality: int):
        name = f"{tag}.jpg"
        cv2.imwrite(str(root / "images" / name), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        (root / "labels" / f"{tag}.txt").write_text("\n".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}" for a, b, c, d in boxes), encoding="utf-8")
        lists[key].append((root / "images" / name).as_posix())
        meta[name] = info

    for key, n, sprites, seed in (("synth_ontoum", args.n_ontoum, sp_on, SEED), ("synth_boos", args.n_boos, sp_boos, SEED + 1),
                                  ("synth_val", args.n_val, sp_on, SEED + 2)):
        rng = np.random.default_rng(seed)
        for i in range(n):
            img, boxes, info = render(rng, sprites, real_bgs)
            write(key, f"{key}_{i:05d}", img, boxes, info | {"set": key}, int(rng.integers(60, 96)))
    for key, rows in (("real_boos", boos_real), ("negatives", negatives)):
        for r in rows:
            src = data.root / r["path"]
            tag = key + "_" + hashlib.sha1(r["path"].encode("utf-8")).hexdigest()[:14]
            shutil.copyfile(src, root / "images" / (tag + src.suffix.lower()))
            boxes = read_yolo_boxes(src) if key == "real_boos" else []
            (root / "labels" / f"{tag}.txt").write_text("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, cx, cy, w, h in boxes), encoding="utf-8")
            lists[key].append((root / "images" / (tag + src.suffix.lower())).as_posix())
            meta[tag + src.suffix.lower()] = {"set": key, "path": r["path"], "group": r["group"], "boxes": len(boxes)}
    exps = {"holdout_boos": ("synth_ontoum", "negatives"), "final": ("synth_ontoum", "synth_boos", "real_boos", "negatives")}
    for exp, keys in exps.items():
        (root / "lists" / f"{exp}_train.txt").write_text("\n".join(p for k in keys for p in lists[k]) + "\n", encoding="utf-8")
        (root / "lists" / f"{exp}_val.txt").write_text("\n".join(lists["synth_val"]) + "\n", encoding="utf-8")
        (root / f"{exp}.yaml").write_text(
            f"path: {root.as_posix()}\ntrain: lists/{exp}_train.txt\nval: lists/{exp}_val.txt\nnames:\n  0: bean\n", encoding="utf-8")
    scenes = {}
    for v in meta.values():
        if "scene" in v:
            s = scenes.setdefault(v["set"] + ":" + v["scene"], {"images": 0, "beans": 0, "max": 0})
            s["images"] += 1; s["beans"] += v["visible_ge_half"]; s["max"] = max(s["max"], v["visible_ge_half"])
    summary = {"name": args.name, "seed": SEED, "size": SIZE, "sprites": {"ontoum224": len(sp_on), "rf_boos": len(sp_boos)},
               "sprite_roasts": {s: {r: sum(x["roast"] == r for x in sp) for r in ROASTS} for s, sp in (("ontoum224", sp_on), ("rf_boos", sp_boos))},
               "sets": {k: len(v) for k, v in lists.items()}, "scenes": scenes, "frozen_groups_excluded": sorted(data.frozen),
               "experiments": {e: {"train_sets": list(k), "train_images": sum(len(lists[x]) for x in k), "val_images": len(lists["synth_val"])}
                               for e, k in exps.items()}}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
