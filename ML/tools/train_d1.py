"""Bounded, group-validated MobileNetV3-Small LOSO, trainval only.

--prepare-views uses no torch and never reads frozen-test image bytes.
--prepare-weights explicitly downloads the official torchvision weights.
Run directories are immutable; current and all reference results are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roastml.decode import decode_image
from roastml.paths import data_dir
from roastml.rgb_views import RGBViewConfig, rgb_view_images
from roastml.segment import lin_to_srgb, srgb_to_lin
from tools.compare_tabular import TEMPERATURES, calibration, calibrate_prob, inner_split, nll, softmax
from tools.train_baseline import CLASSES, FOLDS, SEED, fold_table, load_rows, metrics, video_cap_mask

ML_DIR = Path(__file__).resolve().parents[1]
SEEDS = [20261006, 20261007, 20261008]
CFG = RGBViewConfig()
PLAN = {
    "search_seed": SEED, "confirmation_seeds": SEEDS, "outer_folds": FOLDS,
    "training_cap_seed": SEED, "cap_frames_per_video": 10,
    "inner_group_guard": "QA preregistered near-duplicate candidate connected components; manifest groups and cap unchanged",
    "classes": CLASSES, "rgb_view_config": CFG.to_dict(),
    "training_views": "identical inference-capable full+smart RGB; no annotation bbox crops",
    "agtron": "mandatory EXIF-oriented ROI; no white balance; never see number labels",
    "frozen_head_grid": [{"lr": .01}, {"lr": .003}], "frozen_max_epochs": 40,
    "finetune_grid": [{"lr": .0003, "augmentation": "none"}, {"lr": .0003, "augmentation": "mild"}],
    "finetune_max_epochs": 4, "patience": 2, "batch_images": 32,
    "selection": "inner group validation macro-F1 then NLL then earliest epoch/grid order",
    "temperature_grid": TEMPERATURES, "low_conf_threshold": 0.0,
    "augmentation_mild": {"rotation_degrees": [-180, 180], "horizontal_vertical_flip": True,
                          "crop_scale": [.85, 1.0], "exposure_EV": [-.3, .3],
                          "linear_red_blue_gain_log": [-.08, .08], "noise_sigma": .005,
                          "blur_probability": .15, "JPEG_probability": .2, "JPEG_quality": [70, 95]},
    "forbidden_augmentation": ["hue rotation", "strong saturation", "grayscale", "equalization"],
    "qa_rule": "training-only hard exclusions from preregistered QA; holdouts unchanged",
    "equivalent_qa_arm": "when masks identical, reuse same trained model/probabilities and record equality",
    "limits": "LOSO source-level convenience dataset; no Pi/E2E/test claim; DINOv2 omitted",
}

# Second, separately predeclared protocol (written 2026-10-08 ~20:30, after the
# groupguard run): spec section 6 asks for a none/mild/medium photometric ablation;
# groupguard only covered none/mild. Fine-tune only; everything else unchanged.
AUG_MEDIUM = {"rotation_degrees": [-180, 180], "horizontal_vertical_flip": True,
              "crop_scale": [.5, 1.0], "exposure_EV": [-.5, .5],
              "linear_red_blue_gain_log": [-.2, .2], "noise_sigma": .01,
              "blur_probability": .3, "JPEG_probability": .5, "JPEG_quality": [30, 95]}
PLAN_MEDAUG = {**PLAN,
               "protocol": "medaug_v1",
               "inner_group_guard": PLAN["inner_group_guard"],
               "finetune_grid": [{"lr": .0003, "augmentation": "medium"}, {"lr": .0001, "augmentation": "medium"}],
               "finetune_max_epochs": 6, "augmentation_medium": AUG_MEDIUM,
               "families": ["D1_finetune"],
               "comparison": "same views, cap, inner groups, outer folds and seeds as groupguard; only fine-tune grid/augmentation/epochs differ"}
PROTOCOLS = {
    "groupguard": {"plan": PLAN, "file": "d1_predeclared_20261008_groupguard.json", "run_dir": "d1_20261008_groupguard"},
    "medaug": {"plan": PLAN_MEDAUG, "file": "d1_predeclared_20261008_medaug.json", "run_dir": "d1_20261008_medaug"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def view_job(job):
    path, row, cfg = job
    d = decode_image(Path(path).read_bytes())
    images, diagnostic = rgb_view_images(d, cfg, source=row["source"], roi=row.get("roi", ""))
    return np.asarray(images, np.uint8), diagnostic


def prepare_views(root: Path, workers: int = 4):
    rows = load_rows(root)
    dependency_hashes = {name: sha256(ML_DIR/"roastml"/name) for name in ("rgb_views.py", "decode.py", "segment.py")}
    key = hashlib.sha256(json.dumps({"rows": [(r["path"], r["md5"], r["roi"]) for r in rows],
                                    "config": CFG.to_dict(), "code": dependency_hashes}, sort_keys=True).encode()).hexdigest()[:16]
    # Derived pixels stay only under contact_sheets as requested; cache has
    # numerical embeddings/metadata only. Never put photos inside the repo.
    pixels_dir = root / "contact_sheets" / f"d1_rgb_{key}"
    meta = root / "cache" / f"d1_rgb_{key}.json"
    pixels = pixels_dir / "views.npy"
    if meta.exists() and pixels.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        return rows, np.load(pixels, mmap_mode="r"), info
    pixels_dir.mkdir(parents=True, exist_ok=True)
    temp = pixels_dir / "views.partial.npy"
    arr = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8,
                                   shape=(len(rows), len(CFG.views), CFG.size, CFG.size, 3))
    jobs = [(str(root/r["path"]), r, CFG.to_dict()) for r in rows]
    started = time.perf_counter()
    diagnostics = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for i, (images, diag) in enumerate(ex.map(view_job, jobs, chunksize=8)):
                arr[i] = images; diagnostics.append(diag)
                if (i+1) % 250 == 0: print(f"RGB views {i+1}/{len(rows)}", flush=True)
    else:
        for i, job in enumerate(jobs):
            images, diag = view_job(job); arr[i] = images; diagnostics.append(diag)
    arr.flush(); del arr
    temp.replace(pixels)
    info = {"key": key, "path": str(pixels), "n": len(rows), "shape": [len(rows), 2, 224, 224, 3],
            "manifest_sha256": sha256(root/"manifest.csv"), "rgb_code_sha256": sha256(ML_DIR/"roastml/rgb_views.py"),
            "preprocessing_code_sha256": dependency_hashes,
            "config": CFG.to_dict(), "seconds": time.perf_counter()-started,
            "rows": [{"path": r["path"], "md5": r["md5"], "source": r["source"], "label": r["label"],
                      "group": r["group"], "roi": r["roi"], "diagnostic": d} for r,d in zip(rows,diagnostics)]}
    dump(meta, info)
    return rows, np.load(pixels, mmap_mode="r"), info


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def official_model(pretrained=True):
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    torch.hub.set_dir(str(ML_DIR/"models/pretrained"))
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = mobilenet_v3_small(weights=weights)
    # features+global-average-pool is the backbone; a single new 3-way head.
    net.classifier = torch.nn.Linear(net.classifier[0].in_features, 3)
    return net


def weight_provenance():
    from torchvision.models import MobileNet_V3_Small_Weights
    w = MobileNet_V3_Small_Weights.DEFAULT
    p = ML_DIR/"models/pretrained/checkpoints"/Path(w.url).name
    return {"enum": str(w), "url": w.url, "file": str(p), "sha256": sha256(p),
            "bytes": p.stat().st_size, "metadata": {k:v for k,v in w.meta.items() if k != "categories"},
            "source": "torchvision official model weights; torchvision source BSD-3-Clause; dataset/weights restrictions must remain documented"}


def normalise(images):
    x = np.asarray(images, np.float32) / 255
    return np.ascontiguousarray(((x-np.asarray(CFG.mean,np.float32))/np.asarray(CFG.std,np.float32)).transpose(0,1,4,2,3))


def augment_one(image: np.ndarray, rng, params: dict | None = None) -> np.ndarray:
    """Geometry + camera perturbations, all parameters predeclared (default = mild).

    WB gains act on red/blue in linear RGB and exposure is a linear-light scale:
    no hue rotation, saturation jitter, grayscale or equalisation.
    """
    p = params or PLAN["augmentation_mild"]
    x = image.copy(); n = x.shape[0]
    angle = rng.uniform(*p["rotation_degrees"])
    matrix = cv2.getRotationMatrix2D(((n-1)/2,(n-1)/2),float(angle),1)
    x = cv2.warpAffine(x,matrix,(n,n),borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < .5: x = x[:,::-1]
    if rng.random() < .5: x = x[::-1]
    side = round(n * rng.uniform(*p["crop_scale"])); side = max(8,side)
    a,b = rng.integers(0,n-side+1,size=2)
    x = cv2.resize(x[a:a+side,b:b+side],(n,n),interpolation=cv2.INTER_LINEAR)
    lin = srgb_to_lin(x.astype(np.float32)/255)
    gains = np.exp(rng.uniform(*p["linear_red_blue_gain_log"],3)); gains[1] = 1
    lin = np.clip(lin * gains * 2**rng.uniform(*p["exposure_EV"]),0,1)
    f = lin_to_srgb(lin)
    f = np.clip(f + rng.normal(0,p["noise_sigma"],f.shape),0,1)
    x = np.rint(f*255).astype(np.uint8)
    if rng.random() < p["blur_probability"]: x = cv2.GaussianBlur(x,(3,3),float(rng.uniform(.3,.9)))
    if rng.random() < p["JPEG_probability"]:
        lo, hi = p["JPEG_quality"]
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(x,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(lo,hi+1))])
        if ok: x = cv2.cvtColor(cv2.imdecode(data,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(x)


def augmentation_params(plan: dict, name: str) -> dict | None:
    if name == "none":
        return None
    key = f"augmentation_{name}"
    if key not in plan:
        raise ValueError(f"augmentation '{name}' is not predeclared in this protocol")
    return plan[key]


def embeddings(root, images, info, device):
    import torch
    provenance = weight_provenance()
    key = info["key"] + "_" + provenance["sha256"][:12]
    p = root/"cache"/f"d1_embeddings_{key}.npy"
    if p.exists(): return np.load(p), provenance
    model = official_model().to(device).eval()
    out = []
    with torch.inference_mode():
        for start in range(0,len(images),32):
            x = torch.from_numpy(normalise(images[start:start+32])).to(device)
            b,v,c,h,w = x.shape
            e = model.avgpool(model.features(x.reshape(b*v,c,h,w))).flatten(1).reshape(b,v,-1).mean(1)
            out.append(e.cpu().numpy())
            if start % 512 == 0: print(f"frozen embeddings {start}/{len(images)}",flush=True)
    arr = np.concatenate(out).astype(np.float32)
    np.save(p,arr); del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return arr, provenance


def class_weights(y):
    counts = np.bincount(y,minlength=3)
    if (counts==0).any(): raise ValueError("training split missing class")
    return (len(y)/(3*counts)).astype(np.float32)


def train_head(X, y, cfg, seed, epochs, device, validation=None, plan=PLAN):
    import torch
    seed_all(seed)
    model = torch.nn.Linear(X.shape[1],3).to(device)
    optimiser = torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=.01)
    tx = torch.tensor(np.asarray(X),device=device); ty = torch.tensor(y,dtype=torch.long,device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y),device=device))
    trace=[]; best=None; best_rank=(-1,-float("inf")); patience=0
    for epoch in range(1,epochs+1):
        model.train(); optimiser.zero_grad(); loss=criterion(model(tx),ty); loss.backward(); optimiser.step()
        row={"epoch":epoch,"train_loss":float(loss.detach().cpu())}
        if validation is not None:
            vx,vy=validation; model.eval()
            with torch.inference_mode(): p=softmax(model(torch.tensor(np.asarray(vx),device=device)).cpu().numpy())
            mt=metrics(np.array(CLASSES)[vy],np.array(CLASSES)[p.argmax(1)])
            row.update({"macro_f1":mt["macro_f1"],"nll":nll(np.array(CLASSES)[vy],p)})
            rank=(row["macro_f1"],-row["nll"])
            if rank>best_rank: best_rank=rank; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
            else: patience+=1
        trace.append(row)
        if validation is not None and patience>=plan["patience"]: break
    if validation is not None:
        model.load_state_dict(best)
        selected=max(trace,key=lambda r:(r["macro_f1"],-r["nll"],-r["epoch"]))
    else: selected={"epoch":epochs}
    return model,trace,selected


def predict_network(model, images, ids, device, temperature=1):
    import torch
    model.eval(); out=[]
    with torch.inference_mode():
        for start in range(0,len(ids),32):
            x=torch.from_numpy(normalise(images[ids[start:start+32]])).to(device)
            b,v,c,h,w=x.shape
            logits=model(x.reshape(b*v,c,h,w)).reshape(b,v,3).mean(1)
            out.append(softmax(logits.cpu().numpy()/temperature))
    return np.concatenate(out)


def train_finetune(images,y,ids,cfg,seed,epochs,device,validation=None,plan=PLAN):
    import torch
    seed_all(seed)
    net=official_model().to(device)
    optimiser=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=.01)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y[ids]),device=device))
    rng=np.random.default_rng(seed); trace=[]; best=None; best_rank=(-1,-float("inf")); patience=0
    aug=augmentation_params(plan,cfg["augmentation"])
    started=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train()
        # Freeze BN running statistics with tiny domain-shifted datasets. Affine
        # parameters and all convolution weights still fine-tune normally.
        for module in net.modules():
            if isinstance(module,torch.nn.BatchNorm2d): module.eval()
        loss_sum=0
        shuffled=rng.permutation(ids)
        for start in range(0,len(ids),32):
            batch=shuffled[start:start+32]; pixels=np.array(images[batch],copy=True)
            if aug is not None:
                for i in range(len(pixels)):
                    for j in range(len(CFG.views)): pixels[i,j]=augment_one(pixels[i,j],rng,aug)
            x=torch.from_numpy(normalise(pixels)).to(device); b,v,c,h,w=x.shape
            target=torch.tensor(y[batch],dtype=torch.long,device=device)
            optimiser.zero_grad(); logits=net(x.reshape(b*v,c,h,w)).reshape(b,v,3).mean(1)
            loss=criterion(logits,target); loss.backward(); optimiser.step()
            loss_sum+=float(loss.detach().cpu())*len(batch)
        row={"epoch":epoch,"train_loss":loss_sum/len(ids),"elapsed_seconds":time.perf_counter()-started}
        if validation is not None:
            p=predict_network(net,images,validation,device)
            mt=metrics(np.array(CLASSES)[y[validation]],np.array(CLASSES)[p.argmax(1)])
            row.update({"macro_f1":mt["macro_f1"],"nll":nll(np.array(CLASSES)[y[validation]],p)})
            rank=(row["macro_f1"],-row["nll"])
            if rank>best_rank: best_rank=rank; best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; patience=0
            else: patience+=1
        trace.append(row); print(f"fine {seed} {cfg} epoch {epoch}/{epochs} {row}",flush=True)
        if validation is not None and patience>=plan["patience"]: break
    if validation is not None:
        net.load_state_dict(best)
        selected=max(trace,key=lambda r:(r["macro_f1"],-r["nll"],-r["epoch"]))
    else: selected={"epoch":epochs}
    return net,trace,selected


def choose_temperature(y, p):
    losses=[nll(np.array(CLASSES)[y],calibrate_prob(p,t)) for t in TEMPERATURES]
    i=min(range(len(losses)),key=lambda i:(losses[i],abs(TEMPERATURES[i]-1)))
    return TEMPERATURES[i],{str(t):v for t,v in zip(TEMPERATURES,losses)}


def run(root,run_dir,mask_path,images,rows,info,device,mode,inner_groups_path,plan=PLAN,arms=("reference","qa")):
    import torch,torchvision
    started=time.perf_counter()
    run_dir.mkdir(parents=True,exist_ok=False); dump(run_dir/"plan.json",plan)
    y=np.array([CLASSES.index(r["label"]) for r in rows]); labels=np.array(CLASSES)[y]
    source=np.array([r["source"] for r in rows]); groups=np.array([r["group"] for r in rows]); paths=np.array([r["path"] for r in rows])
    group_info=json.loads(inner_groups_path.read_text(encoding="utf-8"))
    mapping=group_info["group_to_inner_group"]
    if set(groups)-set(mapping): raise ValueError("QA inner group mapping does not cover every training group")
    validation_groups=np.array([mapping[g] for g in groups])
    cap=video_cap_mask(groups,paths,seed=SEED)
    mask=json.loads(mask_path.read_text(encoding="utf-8"))
    if "exclusions" in mask: mask=mask["exclusions"]
    if "mask" in mask: mask=mask["mask"]
    excluded={k for k,v in mask.items() if v is True or (isinstance(v,dict) and v.get("hard_exclude"))}
    clean=cap & np.array([p not in excluded for p in paths])
    dump(run_dir/"training_masks.json",{"reference_rows":int(cap.sum()),"qa_rows":int(clean.sum()),
         "reference_paths":paths[cap].tolist(),"qa_paths":paths[clean].tolist(),"exclusions":sorted(excluded),
         "qa_equal_reference":bool(np.array_equal(cap,clean)),"qa_mask_sha256":sha256(mask_path)})
    E,provenance=embeddings(root,images,info,device)
    versions={"python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__,"torch":torch.__version__,
              "torchvision":torchvision.__version__,"cuda":torch.version.cuda,"device":str(device)}
    families=["D1_frozen","D1_finetune"] if mode=="all" else ["D1_"+mode]
    report={"plan":plan,"versions":versions,"weights":provenance,"data":info,"arms":{},"seconds":None,
            "inner_groups_sha256":sha256(inner_groups_path),"training_code_sha256":sha256(Path(__file__))}
    records=[]
    for family in families:
        for arm,training in [(a,m) for a,m in [("reference",cap),("qa",clean)] if a in arms]:
            if arm=="qa" and np.array_equal(cap,clean):
                report["arms"][family+"_qa"]={"identical_to":family+"_reference","reason":"QA hard exclusions remove no R2 training rows"}
                continue
            all_seeds=[]; choices={}
            for seed in SEEDS:
                probs=np.zeros((len(rows),3)); fold_runs={}
                for fold in FOLDS:
                    tr=np.flatnonzero(training & (source!=fold)); te=np.flatnonzero(source==fold)
                    it,iv,split=inner_split(labels[tr],validation_groups[tr],SEED)
                    ti,vi=tr[it],tr[iv]
                    choice_key=family+"_"+fold
                    if seed==SEED:
                        trials=[]
                        grid=plan["frozen_head_grid"] if family=="D1_frozen" else plan["finetune_grid"]
                        for cfg in grid:
                            if family=="D1_frozen":
                                model,trace,selected=train_head(E[ti],y[ti],cfg,seed,plan["frozen_max_epochs"],device,(E[vi],y[vi]),plan)
                                with torch.inference_mode(): vp=softmax(model(torch.tensor(E[vi],device=device)).cpu().numpy())
                            else:
                                model,trace,selected=train_finetune(images,y,ti,cfg,seed,plan["finetune_max_epochs"],device,vi,plan)
                                vp=predict_network(model,images,vi,device)
                            temperature,losses=choose_temperature(y[vi],vp)
                            trials.append({"config":cfg,"trace":trace,"selected":selected,"temperature":temperature,"temperature_nll":losses})
                            del model
                            if device.type=="cuda": torch.cuda.empty_cache()
                        best=max(range(len(trials)),key=lambda i:(trials[i]["selected"]["macro_f1"],-trials[i]["selected"]["nll"],-i))
                        choices[choice_key]=trials[best]
                        dump(run_dir/f"{family}_{arm}_{fold}_selection.json",{"split":split,"trials":trials,"selected":best,
                             "train_paths":paths[ti].tolist(),"validation_paths":paths[vi].tolist()})
                    chosen=choices[choice_key]; cfg=chosen["config"]; epochs=chosen["selected"]["epoch"]; t=chosen["temperature"]
                    t0=time.perf_counter()
                    if family=="D1_frozen":
                        head,trace,_=train_head(E[tr],y[tr],cfg,seed,epochs,device)
                        with torch.inference_mode(): probs[te]=softmax(head(torch.tensor(E[te],device=device)).cpu().numpy()/t)
                        model=official_model().to(device); model.classifier=head
                    else:
                        model,trace,_=train_finetune(images,y,tr,cfg,seed,epochs,device,plan=plan)
                        probs[te]=predict_network(model,images,te,device,t)
                    checkpoint=ML_DIR/"models/candidates"/run_dir.name/f"{family}_{arm}_{seed}_{fold}"
                    checkpoint.mkdir(parents=True,exist_ok=False)
                    torch.save(model.cpu().state_dict(),checkpoint/"model.pth")
                    fold_runs[fold]={"train_rows":len(tr),"train_groups":len(set(groups[tr])),"holdout_rows":len(te),"holdout_groups":len(set(groups[te])),
                                     "train_paths":paths[tr].tolist(),"config":cfg,"epochs":epochs,"temperature":t,"train_seconds":time.perf_counter()-t0,
                                     "trace":trace,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint/"model.pth")}
                    dump(checkpoint/"model_card.json",{"backend":"torch_training_only","name":checkpoint.name,"family":family,"seed":seed,"fold":fold,
                        "class_order":CLASSES,"rgb_view_config":CFG.to_dict(),"temperature":t,"low_conf_threshold":0.0,"weights":provenance,
                        "versions":versions,"manifest_sha256":info["manifest_sha256"],"config":cfg,"epochs":epochs,
                        "train_rows":len(tr),"train_groups":len(set(groups[tr])),"train_mask_sha256":hashlib.sha256("\n".join(paths[tr]).encode()).hexdigest()})
                    del model
                    if device.type=="cuda": torch.cuda.empty_cache()
                    print(f"completed {family} {arm} seed={seed} fold={fold}",flush=True)
                pred=np.array(CLASSES)[probs.argmax(1)]
                table=fold_table(labels,pred,source)
                entry={"seed":seed,"metrics":table,"calibration":calibration(labels,probs),"fold_runs":fold_runs,
                       "worst_fold":min(FOLDS,key=lambda f:table[f]["macro_f1"])}
                all_seeds.append(entry)
                prediction_rows=[{"path":r["path"],"source":r["source"],"group":r["group"],"label":r["label"],"prediction":p,
                                  **{f"p_{c}":float(q[i]) for i,c in enumerate(CLASSES)}} for r,p,q in zip(rows,pred,probs)]
                csv_write(run_dir/f"{family}_{arm}_{seed}_predictions.csv",prediction_rows)
                records.append({"family":family,"arm":arm,"seed":seed,"mean_macro_f1":table["mean"]["macro_f1"],
                                "pooled_macro_f1":table["pooled"]["macro_f1"],"mean_accuracy":table["mean"]["acc"],
                                "pooled_accuracy":table["pooled"]["acc"],"ece":entry["calibration"]["ece"],
                                "cross_step_rate":table["pooled"]["cross_step_rate"]})
                dump(run_dir/"progress.json",{"records":records,"seconds":time.perf_counter()-started})
            scores=[r["metrics"]["mean"]["macro_f1"] for r in all_seeds]
            report["arms"][family+"_"+arm]={"seeds":all_seeds,"mean_macro_f1_mean":float(np.mean(scores)),"mean_macro_f1_std":float(np.std(scores,ddof=1))}
            dump(run_dir/"report.partial.json",report)
    report["seconds"]=time.perf_counter()-started
    dump(run_dir/"report.json",report); csv_write(run_dir/"compare.csv",records)
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",type=Path); ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--prepare-views",action="store_true"); ap.add_argument("--prepare-weights",action="store_true")
    ap.add_argument("--protocol",choices=sorted(PROTOCOLS),default="groupguard")
    ap.add_argument("--run-dir",type=Path,help="default: ML/results/<protocol run dir>")
    ap.add_argument("--qa-mask",type=Path,default=ML_DIR/"results/qa_audit_20261008/train_exclusion_mask.json")
    ap.add_argument("--inner-groups",type=Path)
    ap.add_argument("--mode",choices=["all","frozen","finetune"],default="all")
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--arms",nargs="+",choices=["reference","qa"],default=["reference","qa"],
                    help="qa only: data ablation against an existing reference run of the same protocol")
    args=ap.parse_args(argv)
    root=args.data_dir.resolve() if args.data_dir else data_dir()
    if args.inner_groups is None: args.inner_groups=root/"cache/qa_audit_20261008/inner_groups.json"
    # Predeclare before data views, weights or any fit. This is a separate plan
    # file, while each immutable run also stores exactly the plan it used.
    protocol=PROTOCOLS[args.protocol]; plan=protocol["plan"]
    if args.run_dir is None: args.run_dir=ML_DIR/"results"/protocol["run_dir"]
    if "families" in plan and args.mode!="finetune" and plan["families"]==["D1_finetune"]:
        raise SystemExit(f"protocol {args.protocol} predeclares fine-tune only; pass --mode finetune")
    plan_path=ML_DIR/"results"/protocol["file"]
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8"))!=json.loads(json.dumps(plan)):
            raise RuntimeError("predeclared D1 plan changed; create a new named protocol")
    else: dump(plan_path,plan)
    if args.prepare_weights:
        official_model(); print(json.dumps(weight_provenance(),ensure_ascii=False),flush=True); return 0
    rows,images,info=prepare_views(root,args.workers)
    if args.prepare_views: print(json.dumps({k:info[k] for k in ["key","n","path","seconds"]}),flush=True); return 0
    if not args.qa_mask.exists(): raise SystemExit("QA audit must complete before model comparison")
    if not args.inner_groups.exists(): raise SystemExit("QA near-duplicate inner-group guard must complete before fitting")
    import torch
    if args.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable; explicitly choose CPU rather than silently changing environment")
    run(root,args.run_dir,args.qa_mask,images,rows,info,torch.device(args.device),args.mode,args.inner_groups,plan,tuple(args.arms))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
