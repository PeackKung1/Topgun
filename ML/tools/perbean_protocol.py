"""Fail-closed development data selection and pre-run declarations for A–E."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

SEED = 20261009
FOLDS = ("ontoum224", "rf_robusta", "rf_boos", "agtron")
GATES = {"flat_mae": 0.5, "touching_mape": 0.10, "pile_mape": 0.25,
         "boos_mae": 0.25, "empty_zero_rate": 0.95, "fold_f1_drop": 0.02,
         "cross_extreme_rate": 0.05, "pi_p95_ms": 500.0}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class DevData:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root / "count_test/count_test_meta.json").read_text(encoding="utf-8"))
        self.frozen = {g for g, split in self.meta["split_existing_by_group"].items() if split == "frozen"}
        manifest = read_csv(self.root / "manifest.csv")
        # Read only metadata for excluded rows. Never open their images/annotations.
        self.rows = [r for r in manifest if r["split"] == "trainval" and r["group"] not in self.frozen]
        self.by_path = {r["path"]: r for r in self.rows}
        self.existing = {r["path"]: r for r in self.meta["existing"]}
        self.count_dev = []
        for r in read_csv(self.root / "count_test/count_test.csv"):
            if r["split"] != "dev":
                continue
            origin = self.existing.get(r["path"])
            if origin:
                if origin["manifest_path"] not in self.by_path:
                    raise ValueError("count dev row is not in safe trainval: " + r["path"])
                r = {**r, "source": origin["source"], "group": origin["group"],
                     "manifest_path": origin["manifest_path"], "label": origin["label"]}
            elif r["path"] in self.by_path:
                r = {**self.by_path[r["path"]], **r}
            else:
                # Unfilled web slots have no image. Filled web rows need explicit
                # provenance; reject unknown images rather than bypassing groups.
                if (self.root / r["path"]).is_file():
                    raise ValueError("web dev image needs manifest trainval group metadata: " + r["path"])
            self.count_dev.append(r)

    def assert_safe(self, row):
        origin = self.existing.get(row["path"])
        p = origin["manifest_path"] if origin else row["path"]
        if p not in self.by_path or row.get("group", self.by_path[p]["group"]) in self.frozen:
            raise ValueError("refusing non-dev/frozen image: " + p)

    def empty_split(self):
        rows = [r for r in self.rows if r["source"] == "rf_hendi" and r["label"].lower() == "empty"]
        groups = sorted({r["group"] for r in rows})
        random.Random(SEED).shuffle(groups)
        tune = set(groups[:len(groups) // 2])
        return ([r for r in rows if r["group"] in tune],
                [r for r in rows if r["group"] not in tune])


def declare(out, name, objective, config, data: DevData):
    """Must be called before opening images or fitting a model in every CLI run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "declaration.json"
    if dest.exists():
        raise FileExistsError("use a new experiment directory; declaration already exists: " + str(dest))
    import cv2
    import numpy as np
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--", "ML"], capture_output=True, check=True).stdout
    ml = Path(__file__).resolve().parent.parent
    source_hashes = {str(p.relative_to(ml)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for package in ('roastml','tools') for p in sorted((ml/package).glob('*.py'))}
    tune, report = data.empty_split()
    declaration = {"name": name, "seed": SEED, "objective": objective, "gates": GATES,
                   "config": config, "revision": revision, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                   "source_sha256": source_hashes,
                   "metadata_sha256": {name:hashlib.sha256((data.root/name).read_bytes()).hexdigest()
                                       for name in ('manifest.csv','count_test/count_test_meta.json','count_test/count_test.csv')},
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
                   "frozen_groups_excluded": sorted(data.frozen), "manifest_test_images_opened": 0,
                   "empty_tune_groups": sorted({r["group"] for r in tune}),
                   "empty_report_groups": sorted({r["group"] for r in report}),
                   "limitations": ["not a Pi latency measurement", "unlabelled count dev scenes cannot pass gates",
                                   "legacy B1 artifacts were trained before the count frozen split; retrain paired fold baselines"]}
    dest.write_text(json.dumps(declaration, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"declared_before_run": str(dest), "objective": objective}, ensure_ascii=False), flush=True)
    return declaration


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
