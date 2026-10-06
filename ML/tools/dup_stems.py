"""ตรวจไฟล์ที่ stem ซ้ำกัน (ชื่อก่อน ".rf.<hash>") ด้วย md5 + pHash — ห้ามตัดสินจากชื่อไฟล์อย่างเดียว

ในแต่ละกลุ่ม stem: จับคู่ไฟล์ทุกคู่ → "ภาพเดียวกัน" ถ้า md5 ตรง หรือ pHash ห่าง ≤ T (ยืนยันซ้ำด้วย mean abs diff ของภาพย่อ)
แล้วรวมภาพเดียวกันเป็น cluster (union-find) · จัดประเภท cluster ที่มี ≥ 2 ไฟล์:
  (ก)  ภาพเดียวกัน อยู่หลาย split (label ตรงกัน)
  (ก2) ภาพเดียวกัน ซ้ำใน split เดียว (label ตรงกัน)
  (ค)  ภาพเดียวกัน แต่ label ต่างกัน = label ขัดกัน → ห้ามใช้ทั้งเทรนและวัดผล (เขียนรายชื่อไฟล์ไว้ใน cache)
กลุ่ม stem ที่มี > 1 cluster = (ข) คนละภาพแต่ชื่อเดียวกัน (นับคู่ที่ label ต่างกันแยกไว้ — ไม่ใช่ label ขัดกัน เพราะคนละภาพ)

ผล: ตัวเลข → ML/results/dup_stems_summary.json · รายชื่อไฟล์ → $ROAST_DATA_DIR/cache/dup_stems/ (ไม่เข้า git)
ใช้: python tools/dup_stems.py [--sources rf_robusta rf_hendi] [--threshold 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from roastml.paths import CACHE, data_dir
from tools.imghash import hamming, load_gray, md5_file, phash_array, thumb_array
from tools.index_sources import Item, index_source

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_T = 6  # pHash Hamming ≤ T = ภาพเดียวกัน (เลือกจาก histogram ระยะในรายงาน ไม่ได้ดูผลโมเดล)
CATS = ("ก_cross_split", "ก2_same_split", "ค_label_conflict")


@dataclass
class Hashed:
    item: Item
    md5: str
    phash: int
    thumb: np.ndarray


def hash_items(root: Path, items: list[Item]) -> list[Hashed]:
    out = []
    for it in items:
        p = root / it.path
        g = load_gray(p)
        out.append(Hashed(it, md5_file(p), phash_array(g), thumb_array(g)))
    return out


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def classify_cluster(members: list[Hashed]) -> str:
    labels = {m.item.label_orig for m in members}
    splits = {m.item.split_orig for m in members}
    if len(labels) > 1:
        return "ค_label_conflict"
    return "ก_cross_split" if len(splits) > 1 else "ก2_same_split"


def analyze(hashed: list[Hashed], threshold: int) -> tuple[dict, list[dict], list[dict]]:
    """คืน (summary, pair_rows, cluster_rows)"""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(hashed):
        groups[h.item.stem].append(i)
    dup_groups = {s: idx for s, idx in groups.items() if len(idx) > 1}

    uf = _UF(len(hashed))
    pair_rows = []
    pair_kind = Counter()
    for stem, idx in dup_groups.items():
        for a, b in combinations(idx, 2):
            ha, hb = hashed[a], hashed[b]
            d = hamming(ha.phash, hb.phash)
            mad = float(np.abs(ha.thumb - hb.thumb).mean())
            ident = "md5" if ha.md5 == hb.md5 else ("phash" if d <= threshold else "different")
            if ident != "different":
                uf.union(a, b)
            same_label = ha.item.label_orig == hb.item.label_orig
            same_split = ha.item.split_orig == hb.item.split_orig
            pair_kind[(ident, "same_split" if same_split else "cross_split",
                       "same_label" if same_label else "diff_label")] += 1
            pair_rows.append({
                "stem": stem, "a": ha.item.path, "b": hb.item.path, "phash_dist": d, "mad64": round(mad, 4),
                "identity": ident, "split_a": ha.item.split_orig, "split_b": hb.item.split_orig,
                "label_a": ha.item.label_orig, "label_b": hb.item.label_orig,
            })

    # cluster ภายในแต่ละกลุ่ม stem
    cluster_rows = []
    cat_clusters, cat_files = Counter(), Counter()
    n_groups_multi_cluster = 0
    for stem, idx in dup_groups.items():
        by_root: dict[int, list[int]] = defaultdict(list)
        for i in idx:
            by_root[uf.find(i)].append(i)
        if len(by_root) > 1:
            n_groups_multi_cluster += 1
        for r, mem in by_root.items():
            if len(mem) < 2:
                continue
            cat = classify_cluster([hashed[i] for i in mem])
            cat_clusters[cat] += 1
            cat_files[cat] += len(mem)
            for i in mem:
                it = hashed[i].item
                cluster_rows.append({"stem": stem, "cluster": f"{stem}#{r}", "category": cat,
                                     "path": it.path, "split_orig": it.split_orig, "label_orig": it.label_orig})

    diff_pairs = [r for r in pair_rows if r["identity"] == "different"]
    nondup = [r["phash_dist"] for r in pair_rows if r["identity"] != "md5"]
    summary = {
        "n_images": len(hashed),
        "n_stems": len(groups),
        "n_stem_groups_dup": len(dup_groups),
        "n_files_in_dup_groups": sum(len(v) for v in dup_groups.values()),
        "group_size_hist": dict(sorted(Counter(len(v) for v in dup_groups.values()).items())),
        "threshold": threshold,
        "pairs_total": len(pair_rows),
        "pairs_by_identity_split_label": {"/".join(k): v for k, v in sorted(pair_kind.items())},
        "clusters_by_category": {c: cat_clusters.get(c, 0) for c in CATS},
        "files_by_category": {c: cat_files.get(c, 0) for c in CATS},
        "ข_stem_groups_with_different_images": n_groups_multi_cluster,
        "ข_pairs_different_image": len(diff_pairs),
        "ข_pairs_different_image_diff_label": sum(r["label_a"] != r["label_b"] for r in diff_pairs),
        # histogram ระยะ pHash ของคู่ที่ md5 ไม่ตรง — ใช้ดูว่า threshold แยกชัดไหม
        "phash_dist_hist_non_md5": dict(sorted(Counter(nondup).items())),
        "mad64_max_same_image": max((r["mad64"] for r in pair_rows if r["identity"] != "different"), default=None),
        "mad64_min_different": min((r["mad64"] for r in diff_pairs), default=None),
        "sensitivity_pairs_same_image_by_T": {
            t: sum(1 for r in pair_rows if r["identity"] == "md5" or r["phash_dist"] <= t) for t in (2, 4, 6, 8, 10, 12)
        },
    }
    return summary, pair_rows, cluster_rows


def global_md5_dups(hashed: list[Hashed]) -> tuple[dict, list[dict]]:
    """ไฟล์ md5 ตรงกันแต่ stem ต่างกัน (ทั้ง source) — ตัดสินได้ชัด ไม่ต้องใช้ threshold
    (pHash ข้าม stem ยังไม่รายงาน: ภาพพื้นเรียบ/เมล็ดเดี่ยวได้ระยะใกล้กันทั้งที่คนละภาพ → ต้อง calibrate ตอนทำ dedupe)
    คืน (summary, แถวของกลุ่มที่ label ขัดกัน)
    """
    by_md5: dict[str, list[Hashed]] = defaultdict(list)
    for h in hashed:
        by_md5[h.md5].append(h)
    groups = [g for g in by_md5.values() if len({h.item.stem for h in g}) > 1]
    conflict_rows = [
        {"md5": h.md5, "path": h.item.path, "split_orig": h.item.split_orig, "label_orig": h.item.label_orig}
        for g in groups if len({h.item.label_orig for h in g}) > 1 for h in g
    ]
    return {
        "md5_groups_other_stem": len(groups),
        "files_in_those_groups": sum(len(g) for g in groups),
        "group_size_hist": dict(sorted(Counter(len(g) for g in groups).items())),
        "groups_cross_split": sum(len({h.item.split_orig for h in g}) > 1 for g in groups),
        "groups_diff_label": sum(len({h.item.label_orig for h in g}) > 1 for g in groups),
        "files_diff_label": len(conflict_rows),
    }, conflict_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--sources", nargs="+", default=["rf_robusta", "rf_hendi"])
    ap.add_argument("--threshold", type=int, default=DEFAULT_T)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = args.data_dir.resolve() if args.data_dir else data_dir()
    out_dir = root / CACHE / "dup_stems"
    report = {}
    for s in args.sources:
        hashed = hash_items(root, index_source(root, s))
        summary, pairs, clusters = analyze(hashed, args.threshold)
        summary["md5_dup_other_stem"], md5_conflicts = global_md5_dups(hashed)
        report[s] = summary
        _write_csv(out_dir / f"{s}_pairs.csv", pairs)
        _write_csv(out_dir / f"{s}_clusters.csv", clusters)
        _write_csv(out_dir / f"{s}_label_conflict.csv", [r for r in clusters if r["category"] == "ค_label_conflict"])
        _write_csv(out_dir / f"{s}_md5_label_conflict_other_stem.csv", md5_conflicts)
        print(f"\n=== {s}")
        print(json.dumps(summary, ensure_ascii=False, indent=1))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "dup_stems_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {out}\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
