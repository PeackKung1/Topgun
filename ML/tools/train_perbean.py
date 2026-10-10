"""Nested source LOSO: paired B1 broadcast / B-bean / B-bean+group.

Only safe trainval single-roast images are used for fitting. Every bean is
weighted 1/n per image, including StandardScaler fitting and validation.
Actual raster synthetic mixtures are made from training-fold bean cutouts
to select grouping thresholds; they are never count acceptance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from roastml.counter import CountConfig
from roastml.features import FEATURES_ALL, FEATURE_SETS, box_features, pixel_stats, select
from roastml.linear_model import LinearSoftmax, spec_from_sklearn
from roastml.paths import data_dir
from roastml.perbean import bean_features, group_predictions
from roastml.segment import SegConfig, prepare_image, segment
from tools.index_sources import read_yolo_boxes
from tools.perbean_protocol import DevData, FOLDS, GATES, SEED, declare, dump
from tools.tune_counter import load_rgb

CLASSES = ["light", "medium", "dark"]
GRID = [0.01, 0.1, 1.0, 10.0]
GROUP_GAPS = [4.0, 8.0, 12.0, 16.0, 24.0]
NAMES = FEATURE_SETS["Lab_hist"]
_data = None


def worker_init(root):
    global _data
    _data = DevData(root)
    cv2.setNumThreads(1)


def extract(job):
    r, spec = job
    rgb = load_rgb(_data, r)
    sc, cc = SegConfig(), CountConfig.from_dict(spec)
    prep = prepare_image(rgb, sc, allow_wb=r["source"] != "agtron")
    seg = segment(rgb, sc, find_beans=r["source"] != "agtron", prepared=prep)
    result, X = bean_features(rgb, cc, sc, force_pile=r["source"] == "agtron", prepared=prep)
    image_x = pixel_stats(seg.lab[seg.pixel_mask])
    train_x = image_x
    if r["source"] in ("rf_boos", "rf_robusta"):
        boxes = read_yolo_boxes(_data.root/r["path"])
        if boxes:
            train_x = box_features(rgb, [b[1:] for b in boxes], sc).x
    return {"row": r, "X": X, "image_x": image_x, "train_x": train_x,
            "n": result.n, "scene": result.scene, "method": result.method}


def weighted_metrics(y, pred, weight):
    y, pred, weight = np.asarray(y), np.asarray(pred), np.asarray(weight, float)
    cm = np.zeros((3, 3), float)
    for i,a in enumerate(CLASSES):
        for j,b in enumerate(CLASSES):
            cm[i,j] = weight[(y == a) & (pred == b)].sum()
    f1 = np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1), out=np.zeros(3), where=(cm.sum(0)+cm.sum(1)) > 0)
    extreme = cm[0].sum()+cm[2].sum()
    return {"macro_f1": float(f1.mean()), "f1_per_class": dict(zip(CLASSES, f1.tolist())),
            "light_dark_rate": float((cm[0,2]+cm[2,0])/extreme) if extreme else None,
            "image_weight_sum": float(weight.sum()), "beans": len(y), "cm": cm.tolist()}


def arrays(records):
    records = [r for r in records if r["n"]]
    if not records:
        raise ValueError("no bean features in train fold")
    X = np.concatenate([r["X"] for r in records])
    y = np.concatenate([np.repeat(r["row"]["label"], r["n"]) for r in records])
    w = np.concatenate([np.full(r["n"], 1/r["n"]) for r in records])
    return X, y, w


def fit_linear(X, y, w, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))])
    pipe.fit(select(X, NAMES), y, sc__sample_weight=w, lr__sample_weight=w)
    return spec_from_sklearn(pipe, NAMES, CLASSES)


def choose_C(records):
    scores = {}
    sources = sorted({r["row"]["source"] for r in records})
    for C in GRID:
        inner = []
        for hold in sources:
            tr = [r for r in records if r["row"]["source"] != hold]
            te = [r for r in records if r["row"]["source"] == hold and r["n"]]
            X,y,w = arrays(tr)
            model = LinearSoftmax(fit_linear(X,y,w,C))
            Xt,yt,wt = arrays(te)
            pp = np.array(CLASSES)[model.proba(Xt).argmax(1)]
            inner.append(weighted_metrics(yt,pp,wt)["macro_f1"])
        scores[str(C)] = float(np.mean(inner))
    best = max(GRID, key=lambda C: (scores[str(C)], -C))
    return best, scores


def fit_broadcast(records):
    # Reproduce B1 R2's 10-frame video cap. No legacy fitted artifact is read.
    rng = np.random.default_rng(20261006)
    keep = []
    for group in sorted({r["row"]["group"] for r in records}):
        rr = sorted([r for r in records if r["row"]["group"] == group], key=lambda r: r["row"]["path"])
        if ":video:" in group and len(rr) > 10:
            rr = [rr[i] for i in sorted(rng.choice(len(rr), 10, replace=False))]
        keep += rr
    from tools.train_baseline import fit_b1
    return fit_b1(np.stack([r["train_x"] for r in keep]), np.array([r["row"]["label"] for r in keep]), "Lab_hist", .01)


def synthetic_mixtures(data, records, cc, model, cache_dir):
    """Render cutouts from train rows, label each detected core by pasted truth."""
    rng = np.random.default_rng(SEED)
    sprites = {c: [] for c in CLASSES}
    sc = SegConfig()
    for c in CLASSES:
        # Select source-diverse cutouts from available positive detections.
        pool = sorted([r for r in records if r["row"]["label"] == c and r["n"]], key=lambda r: (r["row"]["source"],r["row"]["path"]))
        picks = rng.choice(len(pool), min(12,len(pool)), replace=False)
        for idx in picks:
            row = pool[idx]["row"]
            rgb = load_rgb(data,row)
            prep = prepare_image(rgb,sc,allow_wb=row["source"] != "agtron")
            res,_ = bean_features(rgb,cc,sc,force_pile=row["source"] == "agtron",prepared=prep)
            if not res.n:
                continue
            areas = np.bincount(res.labels.ravel())[1:]
            bid = int(areas.argmax())+1
            yy,xx = np.nonzero(res.labels == bid)
            mask = res.labels[yy.min():yy.max()+1,xx.min():xx.max()+1] == bid
            patch = (prep.wb.f[yy.min():yy.max()+1,xx.min():xx.max()+1]*255).astype(np.uint8)
            factor = 56/max(patch.shape[:2])
            size = (max(1,round(patch.shape[1]*factor)), max(1,round(patch.shape[0]*factor)))
            sprites[c].append((cv2.resize(patch,size),cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST).astype(bool)))
        if not sprites[c]:
            raise ValueError("no synthetic train sprite for " + c)
    cache_dir.mkdir(parents=True,exist_ok=True)
    samples = []
    # Balanced 1/2/3-class scenes; single-roast scenes penalize false splitting.
    for j in range(48):
        k = j%3+1
        present = rng.choice(3,k,replace=False)
        image = np.full((256,320,3),235,np.uint8)
        truth = np.full((256,320),-1,np.int32)
        for i in range(9):
            cls = int(present[i%k])
            patch,mask = sprites[CLASSES[cls]][rng.integers(len(sprites[CLASSES[cls]]))]
            h,w = mask.shape
            x,y = 25+(i%3)*100,20+(i//3)*78
            image[y:y+h,x:x+w][mask] = patch[mask]
            truth[y:y+h,x:x+w][mask] = cls
        # Images belong in the external data cache; never in git/results.
        cv2.imwrite(str(cache_dir/f"mix_{j:02d}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        res,X = bean_features(image,cc,sc)
        target = []
        for i in range(1,res.n+1):
            v = truth[res.core & (res.labels == i)]
            v = v[v >= 0]
            target.append(int(np.bincount(v,minlength=3).argmax()) if len(v) else -1)
        target = np.asarray(target)
        ok = target >= 0
        if ok.any():
            samples.append((X[ok],model.proba(X[ok]),np.array(CLASSES)[target[ok]], k))
    return samples


def choose_group(samples):
    scores = {}
    for gap in GROUP_GAPS:
        yt,yp,ww = [],[],[]
        false_mixed = 0
        for X,P,y,k in samples:
            idx,_,info = group_predictions(X,P,CLASSES,{"min_L_gap":gap,"max_k":3})
            yt.extend(y); yp.extend(np.array(CLASSES)[idx]); ww.extend(np.full(len(y),1/len(y)))
            false_mixed += int(k == 1 and info["k"] > 1)
        scores[str(gap)] = {**weighted_metrics(yt,yp,ww), "false_mixed_single_scenes":false_mixed}
    gap = max(GROUP_GAPS, key=lambda g: (scores[str(g)]["macro_f1"],-scores[str(g)]["false_mixed_single_scenes"],g))
    return {"min_L_gap":gap,"max_k":3}, scores


def evaluate(records,b1,bean,group):
    yt,ww,raw = [],[],[]
    predictions = {name: [] for name in ("B1 broadcast","B-bean","B-bean+group")}
    for r in records:
        if not r["n"]:
            continue
        n=r["n"]; X=r["X"]
        P=bean.proba(X)
        image_pred = CLASSES[int(b1.proba(r["image_x"])[0].argmax())]
        pred = np.array(CLASSES)[P.argmax(1)]
        grouped,_,info=group_predictions(X,P,CLASSES,group)
        yt.extend([r["row"]["label"]]*n); ww.extend([1/n]*n)
        predictions["B1 broadcast"].extend([image_pred]*n)
        predictions["B-bean"].extend(pred)
        predictions["B-bean+group"].extend(np.array(CLASSES)[grouped])
        raw.append({"path":r["row"]["path"],"n":n,"label":r["row"]["label"],"B1":image_pred,
                    "bean_counts":{c:int(np.sum(pred == c)) for c in CLASSES}, "group_k":info["k"]})
    metrics = {name:{"image_weighted":weighted_metrics(yt,pred,ww),"unweighted":weighted_metrics(yt,pred,np.ones(len(yt)))}
               for name,pred in predictions.items()}
    return metrics,raw


def run_mixed(data,records,folds,cc):
    b1=LinearSoftmax(folds["rf_boos"]["b1"])
    bean=LinearSoftmax(folds["rf_boos"]["bean"])
    group=folds["rf_boos"]["group"]
    output=[]
    from tools.index_sources import read_yolo_names
    names=read_yolo_names(data.root/"raw/rf_boos/data.yaml")
    mapping=[next(c for c in CLASSES if c in n.lower()) for n in names]
    for r in records:
        row=r["row"]
        if row["source"] != "rf_boos" or row["label"] != "mixed":
            continue
        boxes=read_yolo_boxes(data.root/row["path"])
        gt=np.bincount([CLASSES.index(mapping[b[0]]) for b in boxes],minlength=3)
        n=r["n"]
        base=int(b1.proba(r["image_x"])[0].argmax())
        counts={"B1 broadcast":np.eye(3,dtype=int)[base]*n}
        if n:
            P=bean.proba(r["X"])
            gg,_,_=group_predictions(r["X"],P,CLASSES,group)
            counts["B-bean"]=np.bincount(P.argmax(1),minlength=3)
            counts["B-bean+group"]=np.bincount(gg,minlength=3)
        else:
            counts.update({"B-bean":np.zeros(3,int),"B-bean+group":np.zeros(3,int)})
        output.append({"path":row["path"],"gt":gt.tolist(),"n":n,
                       "counts":{k:v.tolist() for k,v in counts.items()},
                       "errors":{k:int(np.abs(v-gt).sum()) for k,v in counts.items()}})
    dev_paths={r["path"] for r in data.count_dev if r.get("n_total")}
    dev=[r for r in output if r["path"] in dev_paths]
    def aggregate(rows):
        from tools.tune_counter import bootstrap
        return {name:bootstrap([r["errors"][name] for r in rows]) for name in ("B1 broadcast","B-bean","B-bean+group")}
    return {"dev":aggregate(dev),"mixed_video":aggregate(output),"paired_dev":dev,"paired_video":output}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--count-config",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--cache",type=Path)
    args=p.parse_args()
    data=DevData(data_dir())
    cc=CountConfig.from_dict(json.loads(args.count_config.read_text(encoding="utf-8")))
    declare(args.out,"per-bean-nested-LOSO",{"C_grid":GRID,"selection":"highest inner source LOSO image-weighted macro-F1; ties smaller C",
            "B1":"refit R2 Lab_hist C=.01, 10 video frames; same outer train sources; no legacy fitted model",
            "group_gaps":GROUP_GAPS,"group_selection":"train-fold raster mixtures macro-F1; tie fewer false mixed then larger gap",
            "group_k":[1,2,3],"weights":"1/n per image in scaler, LR and metrics", "macro_f1_classes":CLASSES,
            "limit_per_source":args.limit},
            {"count_config":cc.to_dict(),"seg_config":SegConfig().to_dict()},data)
    rows=[r for r in data.rows if r["source"] in FOLDS and r["label"] in CLASSES+['mixed']]
    if args.limit:
        rng=np.random.default_rng(SEED)
        rows=[r for s in FOLDS for r in rng.choice([r for r in rows if r["source"] == s],min(args.limit,sum(r["source"] == s for r in rows)),replace=False)]
    feature_sources={name:hashlib.sha256((Path(__file__).resolve().parent.parent/'roastml'/name).read_bytes()).hexdigest()
                     for name in ('counter.py','segment.py','features.py','perbean.py','rgb_views.py','decode.py')}
    fingerprint=hashlib.sha256(json.dumps({"paths":[(r["path"],r.get('md5'),r.get('roi')) for r in rows],
                          "config":cc.to_dict(),"sources":feature_sources,"version":4},sort_keys=True).encode()).hexdigest()[:16]
    cache=data.root/"cache"/f"perbean_features_{fingerprint}.npz"
    meta=cache.with_suffix('.json')
    if args.cache:
        cache=args.cache; meta=cache.with_suffix('.json')
    if cache.is_file() and meta.is_file():
        metadata=json.loads(meta.read_text(encoding='utf-8'))
        if metadata['fingerprint'] != fingerprint:
            raise ValueError('feature cache config/path mismatch')
        z=np.load(cache,allow_pickle=False)
        records=[{**r,'X':z[f'x{i}'],'image_x':z[f'im{i}'],'train_x':z[f'tr{i}']} for i,r in enumerate(metadata['records'])]
    else:
        for row in rows: data.assert_safe(row)
        if args.workers == 1:
            worker_init(data.root); records=[extract((r,cc.to_dict())) for r in rows]
        else:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(data.root,)) as pool:
                records=[]
                for i,r in enumerate(pool.map(extract,[(r,cc.to_dict()) for r in rows],chunksize=8)):
                    records.append(r)
                    if (i+1)%200 == 0: print(f'features {i+1}/{len(rows)}',flush=True)
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,**{f'{k}{i}':r[key] for i,r in enumerate(records) for k,key in [('x','X'),('im','image_x'),('tr','train_x')]})
        dump(meta,{'fingerprint':fingerprint,'records':[{k:v for k,v in r.items() if k not in ('X','image_x','train_x')} for r in records]})
    single=[r for r in records if r['row']['label'] in CLASSES]
    allfolds={}; summary={}; raw=[]
    for hold in FOLDS:
        train=[r for r in single if r['row']['source'] != hold]
        test=[r for r in single if r['row']['source'] == hold]
        C,inner=choose_C(train)
        X,y,w=arrays(train)
        bean_spec=fit_linear(X,y,w,C); bean=LinearSoftmax(bean_spec)
        b1_spec=fit_broadcast(train)
        samples=synthetic_mixtures(data,train,cc,bean,data.root/'cache'/args.out.name/hold)
        group,group_scores=choose_group(samples)
        paired,predictions=evaluate(test,LinearSoftmax(b1_spec),bean,group)
        summary[hold]={'paired':paired,'C':C,'inner_C_scores':inner,'group_config':group,'group_train_synthetic':group_scores,
                       'images':len(test),'zero_detection_images':sum(r['n']==0 for r in test)}
        base=paired['B1 broadcast']['image_weighted']['macro_f1']
        for name in ('B-bean','B-bean+group'):
            score=paired[name]['image_weighted']['macro_f1']
            summary[hold][name+'_noninferiority_pass']=score >= base-GATES['fold_f1_drop']
        allfolds[hold]={'b1':b1_spec,'bean':bean_spec,'group':group}
        raw.extend(predictions)
        dump(args.out/'partial_summary.json',summary)
        print(json.dumps({'fold':hold,'C':C,'group':group,'F1':{n:m['image_weighted']['macro_f1'] for n,m in paired.items()}}),flush=True)
    mixed=run_mixed(data,records,allfolds,cc)
    dump(args.out/'summary.json',{'folds':summary,'mixed':mixed,'feature_cache':str(cache),
         'eligible':False,'limitations':['touching/pile count dev GT missing','Pi not measured','detected beans only; report zero-detection coverage']})
    dump(args.out/'per_image.json',raw)
    modeldir=Path(__file__).resolve().parent.parent/'models'/args.out.name
    modeldir.mkdir(parents=True,exist_ok=True)
    # Fold models stay internal/ignored. No automatic deploy selection.
    for hold,specs in allfolds.items():
        d=modeldir/hold; d.mkdir(exist_ok=True)
        dump(d/'model.json',specs['b1']); dump(d/'bean_model.json',specs['bean'])
        dump(d/'model_card.json',{'backend':'b1_linear_beans','name':f'perbean-dev-fold-{hold}',
             'model_file':'model.json','bean_model_file':'bean_model.json','seg_config':SegConfig().to_dict(),
             'count_config':cc.to_dict(),'group_config':specs['group'],'low_conf_threshold':0.0,
             'train_sources':[s for s in FOLDS if s != hold], 'frozen_groups_excluded':sorted(data.frozen),
             'validation':'experimental; not deployment eligible; no frozen evaluation; unknown Agtron license'})
    print(json.dumps({'model_dir':str(modeldir),'eligible':False}),flush=True)


if __name__ == '__main__':
    main()
