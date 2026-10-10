"""Predeclared counter diagnostics/tuning. Never reads test or frozen groups.

python -m tools.tune_counter --out results/<unique-run> [--configs configs.json]
Tuning uses only the deterministic first half of hendi Empty groups. --report
evaluates the other half after a single config has been fixed, and never selects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np

from roastml.counter import CountConfig, count_beans
from roastml.decode import decode_image
from roastml.paths import data_dir
from roastml.segment import SegConfig, segment
from tools.index_sources import read_yolo_boxes
from tools.perbean_protocol import DevData, SEED, declare, dump

_worker_data = None

def init_worker(root):
    global _worker_data
    import cv2
    cv2.setNumThreads(1)
    _worker_data = DevData(root)

def evaluate_row(job):
    import time
    r,specs=job
    rgb=load_rgb(_worker_data,r)
    cfg=SegConfig()
    gt=0 if r['source']=='rf_hendi' else 1
    if r['source']=='rf_boos': gt=len(read_yolo_boxes(_worker_data.root/r['path']))
    mode=segment(rgb,cfg).mode
    records=[]
    for name,spec in specs.items():
        t=time.perf_counter()
        cr=count_beans(rgb,CountConfig.from_dict(spec),cfg)
        records.append({'path':r['path'],'source':r['source'],'group':r['group'],'config':name,
                        'gt':gt,'n':cr.n,'scene':cr.scene,'method':cr.method,'ref_area':cr.ref_area,
                        'seg_mode':mode,'ms':(time.perf_counter()-t)*1000,'notes':cr.notes})
    return records


def bootstrap(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"n": 0, "value": None, "ci95": None, "status": "unmeasured"}
    rng = np.random.default_rng(SEED)
    means = np.concatenate([v[rng.integers(0, len(v), (min(500, 10000-i), len(v)))].mean(1)
                            for i in range(0, 10000, 500)])
    return {"n": len(v), "value": float(v.mean()), "ci95": np.percentile(means, [2.5, 97.5]).tolist()}


def load_rgb(data, r):
    data.assert_safe(r)
    img = decode_image((data.root / r["path"]).read_bytes())
    rgb = np.asarray(img.image)
    if r.get("roi"):
        from roastml.rgb_views import crop_roi
        rgb = crop_roi(img, r["roi"])
    return rgb


def summarize(records):
    out = {}
    for field, source in (("ontoum", "ontoum224"), ("boos_beans", "rf_boos"), ("empty", "rf_hendi")):
        rr = [r for r in records if r["source"] == source and (source != "rf_boos" or r["seg_mode"] == "beans")]
        metric = bootstrap([float(r["n"] == 0) if field == "empty" else abs(r["n"]-r["gt"]) for r in rr])
        metric["metric"] = "zero_rate" if field == "empty" else "MAE"
        out[field] = metric
    return out


def main():
    import json
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--configs", type=Path)
    p.add_argument("--report", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    data = DevData(args.data_dir or data_dir())
    specs = json.loads(args.configs.read_text(encoding="utf-8")) if args.configs else {"wip": CountConfig().to_dict()}
    configs = {name: CountConfig.from_dict(spec) for name, spec in specs.items()}
    specs = {name:cfg.to_dict() for name,cfg in configs.items()}
    if args.report and len(configs) != 1:
        raise ValueError("report half must evaluate exactly one pre-fixed configuration")
    declare(args.out, "counter-report" if args.report else "counter-tune", {
        "selection": "lexicographic: highest tune-half Empty zero-rate, lowest labeled count-dev flat MAE, lowest boos beans MAE, lowest ontoum MAE; ties earlier config",
        "report_half_used_for_selection": False, "pile": "no tuning without counted dev GT; synthetic evidence only",
        "diagnostics": "ontoum target 1; boos YOLO number of boxes; inspect spikes >20 and overcounted small scenes",
        "limit_per_source": args.limit,"workers":args.workers,
        "visibility":"visible region area >=.5 reference area heuristic; not proof of physical visibility"}, specs, data)
    rows = [r for r in data.rows if r["source"] in ("ontoum224", "rf_boos") and r["label"] in ("light", "medium", "dark", "mixed")]
    empty_tune, empty_report = data.empty_split()
    rows += empty_report if args.report else empty_tune
    if args.limit:
        picked = []
        rng = np.random.default_rng(SEED)
        for s in ("ontoum224", "rf_boos", "rf_hendi"):
            sub = sorted([r for r in rows if r["source"] == s], key=lambda r: r["path"])
            ix = rng.choice(len(sub), min(args.limit, len(sub)), replace=False)
            picked.extend(sub[i] for i in ix)
        rows = picked
    for row in rows: data.assert_safe(row)
    records=[]
    jobs=[(r,specs) for r in rows]
    if args.workers == 1:
        init_worker(data.root)
        results=map(evaluate_row,jobs)
        pool=None
    else:
        pool=ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(data.root,))
        results=pool.map(evaluate_row,jobs,chunksize=8)
    try:
        for j,rr in enumerate(results):
            records.extend(rr)
            if (j+1)%100 == 0:
                dump(args.out/'per_image.partial.json',records)
                print(f'counted {j+1}/{len(rows)}',flush=True)
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
    dump(args.out / "per_image.json", records)
    summary = {name: summarize([r for r in records if r["config"] == name]) for name in configs}
    dev_gt = {r['path']:int(r['n_total']) for r in data.count_dev if r.get('n_total') and r['scene'] == 'flat'}
    for name in configs:
        rr = [r for r in records if r['config'] == name and r['path'] in dev_gt]
        summary[name]['count_dev_flat'] = bootstrap([abs(r['n']-dev_gt[r['path']]) for r in rr])
    if not args.report:
        def score(name):
            s = summary[name]
            return (-(s["empty"]["value"] or 0), s['count_dev_flat']['value'] if s['count_dev_flat']['value'] is not None else float('inf'),
                    s["boos_beans"]["value"] if s["boos_beans"]["value"] is not None else float('inf'),
                    s["ontoum"]["value"] if s["ontoum"]["value"] is not None else float('inf'))
        chosen = min(configs, key=score)
        dump(args.out / "selected_config.json", configs[chosen].to_dict())
    else:
        chosen = next(iter(configs))
    dump(args.out / "summary.json", {"summary": summary, "chosen": chosen, "report_only": args.report,
                                   "count_test_touching": "unmeasured", "count_test_pile": "unmeasured"})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    spikes = [r for r in records if r["source"] == "ontoum224" and r["n"] > 20]
    print(json.dumps({"ontoum_spikes": spikes[:10]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
