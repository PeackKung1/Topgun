import json,sys,collections,numpy as np
sys.path.insert(0,".")
from roastml.linear_model import LinearSoftmax
from roastml.perbean import group_predictions
from roastml.features import FEATURES_ALL
D=r"C:/Users/Public/Documents/TOPGUN_CONTEST/data/"
SW="../../roast-classification-seed-count-91d3d9/ML/"
meta=json.load(open(D+"cache/perbean_features_d804b44428eb274a.json",encoding="utf-8"))
z=np.load(D+"cache/perbean_features_d804b44428eb274a.npz",allow_pickle=False)
recs=[{**r,"X":z[f"x{i}"],"image_x":z[f"im{i}"],"train_x":z[f"tr{i}"]} for i,r in enumerate(meta["records"])]
CL=["light","medium","dark"]; FOLDS=["ontoum224","rf_robusta","rf_boos","agtron"]
cm_meta=json.load(open(D+"count_test/count_test_meta.json",encoding="utf-8"))
frozen={g for g,s in cm_meta["split_existing_by_group"].items() if s=="frozen"}
print("records",len(recs),"splits",collections.Counter(r["row"]["split"] for r in recs),"in frozen groups",sum(r["row"]["group"] in frozen for r in recs))
print("by source/label (images | beans):")
for s in FOLDS:
    print(" ",s,{l:(sum(1 for r in recs if r["row"]["source"]==s and r["row"]["label"]==l), int(sum(r["n"] for r in recs if r["row"]["source"]==s and r["row"]["label"]==l)), sum(1 for r in recs if r["row"]["source"]==s and r["row"]["label"]==l and r["n"]==0)) for l in CL+["mixed"]})
# group leakage
g2s=collections.defaultdict(set)
for r in recs: g2s[r["row"]["group"]].add(r["row"]["source"])
print("groups spanning >1 source:",sum(len(v)>1 for v in g2s.values()), "n groups",len(g2s))
# md5 / phash duplicates across sources
md=collections.defaultdict(set); ph=collections.defaultdict(set)
for r in recs: md[r["row"]["md5"]].add(r["row"]["source"]); ph[r["row"]["phash"]].add(r["row"]["source"])
print("md5 across sources:",sum(len(v)>1 for v in md.values()),"phash across sources:",sum(len(v)>1 for v in ph.values()))
def f1s(y,p,w):
    cm=np.zeros((3,3))
    for yy,pp,ww in zip(y,p,w): cm[yy,pp]+=ww
    f=[2*cm[i,i]/(cm[i].sum()+cm[:,i].sum()) if cm[i].sum()+cm[:,i].sum()>0 else 0 for i in range(3)]
    ext=cm[0].sum()+cm[2].sum()
    return float(np.mean(f)),f,(cm[0,2]+cm[2,0])/ext if ext else None,cm
summ=json.load(open(SW+"results/perbean_final_loso_20261010/summary.json",encoding="utf-8"))["folds"]
single=[r for r in recs if r["row"]["label"] in CL]
pooled={k:np.zeros((3,3)) for k in ("B1 broadcast","B-bean","B-bean+group")}
for hold in FOLDS:
    md_=SW+f"models/perbean_final_loso_20261010/{hold}/"
    card=json.load(open(md_+"model_card.json",encoding="utf-8"))
    b1=LinearSoftmax.load(md_+"model.json"); bean=LinearSoftmax.load(md_+"bean_model.json")
    assert b1.classes==CL and bean.classes==CL,(b1.classes,bean.classes)
    test=[r for r in single if r["row"]["source"]==hold and r["n"]]
    train=[r for r in single if r["row"]["source"]!=hold and r["n"]]
    # scaler check: weighted mean/std of TRAIN beans only vs stored scaler
    X=np.concatenate([r["X"] for r in train])[:,bean.idx]; w=np.concatenate([np.full(r["n"],1/r["n"]) for r in train])
    mu=(X*w[:,None]).sum(0)/w.sum(); sd=np.sqrt((w[:,None]*(X-mu)**2).sum(0)/w.sum())
    Xa=np.concatenate([r["X"] for r in single if r["n"]])[:,bean.idx]; wa=np.concatenate([np.full(r["n"],1/r["n"]) for r in single if r["n"]])
    mua=(Xa*wa[:,None]).sum(0)/wa.sum()
    print(f"== {hold}: scaler vs train-only weighted mean maxabs {np.abs(mu-bean.mean).max():.2e} std {np.abs(sd-bean.scale).max():.2e} | vs all-data mean {np.abs(mua-bean.mean).max():.3f} | card train_sources {card['train_sources']} group {card['group_config']}")
    y=[];w_=[];P1=[];P2=[];P3=[]
    for r in test:
        n=r["n"];P=bean.proba(r["X"]);g,_,_=group_predictions(r["X"],P,CL,card["group_config"])
        ip=int(b1.proba(r["image_x"])[0].argmax())
        y+= [CL.index(r["row"]["label"])]*n; w_+=[1/n]*n; P1+=[ip]*n; P2+=list(P.argmax(1)); P3+=list(g)
    for name,p in (("B1 broadcast",P1),("B-bean",P2),("B-bean+group",P3)):
        f,per,ld,cm=f1s(y,p,w_); fu,_,_,_=f1s(y,p,np.ones(len(y))); pooled[name]+=cm
        ref=summ[hold]["paired"][name]["image_weighted"]; refu=summ[hold]["paired"][name]["unweighted"]["macro_f1"]
        print(f"   {name:13s} F1w {f:.4f} (summary {ref['macro_f1']:.4f}) unw {fu:.4f} ({refu:.4f}) ld {ld:.4f} ({ref['light_dark_rate']:.4f}) beans {len(y)} imgs {len(test)} classes_present {sorted(set(y))} cm_maxdiff {np.abs(cm-np.array(ref['cm'])).max():.2e}")
for name,cm in pooled.items():
    print("pooled",name,"light↔dark", round((cm[0,2]+cm[2,0])/(cm[0].sum()+cm[2].sum()),4), "L→D",round(cm[0,2],1),"D→L",round(cm[2,0],1))
