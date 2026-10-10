"""Predeclared stage latency, visible-count dev and 300-bean notebook proxy.

Never establishes the Pi gate. Sampling/profiling uses only safe dev/trainval
groups. Reports separate decode/WB/segment/count/features/classify/group.
"""
from __future__ import annotations
import argparse
import io
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from roastml.api import load
from roastml.paths import data_dir
from tools.perbean_protocol import DevData, FOLDS, SEED, declare, dump
from tools.tune_counter import bootstrap

STAGES = ('decode','WB','segment','count','features','classify','group','fallback','total')

def synthetic_300():
    image=np.full((600,900,3),235,np.uint8)
    for i in range(300):
        color=((170,125,85),(110,72,45),(50,32,22))[i%3]
        cv2.ellipse(image,(20+(i%20)*44,20+(i//20)*39),(13,9),0,0,360,color,-1)
    return image

def synthetic_pile_300():
    """Painter-order ellipses with pixel ground truth for >=half visibility."""
    image=np.full((600,900,3),235,np.uint8)
    labels=np.zeros((600,900),np.int32)
    full_area=[]
    for i in range(300):
        mask=np.zeros(labels.shape,np.uint8)
        center=(28+(i%20)*44,25+(i//20)*39)
        cv2.ellipse(mask,center,(27,23),0,0,360,1,-1)
        full_area.append(int(mask.sum()))
        color=((170,125,85),(110,72,45),(50,32,22))[i%3]
        image[mask>0]=color
        labels[mask>0]=i+1
        cv2.line(image,(center[0],center[1]-12),(center[0],center[1]+12),tuple(max(0,c-12) for c in color),2)
    visible=np.bincount(labels.ravel(),minlength=301)[1:]/np.asarray(full_area)
    return image,int(np.sum(visible>=.5)),visible.tolist()

def jpeg(rgb):
    out=io.BytesIO(); Image.fromarray(rgb).save(out,'JPEG',quality=85); return out.getvalue()

def percentiles(values):
    v=np.asarray(values,float)
    return {'n':len(v),'p50_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} if len(v) else {'n':0,'p50_ms':None,'p95_ms':None}

def peak_ram():
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                      ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                       'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        m=Counters(); m.cb=ctypes.sizeof(m)
        ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
        getinfo=ctypes.windll.psapi.GetProcessMemoryInfo
        getinfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if getinfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(m),m.cb):
            return {'peak_RSS_MiB':m.PeakWorkingSetSize/2**20,'RSS_MiB':m.WorkingSetSize/2**20}
        return {'peak_RSS_MiB':None,'RSS_MiB':None}
    import resource
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_RSS_MiB':rss/(2**20 if platform.system() == 'Darwin' else 1024)}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeat',type=int,default=3)
    args=p.parse_args()
    if args.repeat < 1: raise ValueError('repeat must be positive')
    data=DevData(data_dir())
    card=json.loads((args.model/'model_card.json').read_text(encoding='utf-8'))
    declare(args.out,'perbean-notebook-benchmark',{'sampling':'50 safe trainval single-roast/source, seed20261009 + all filled dev images',
            'repeat':args.repeat,'stages':STAGES,'synthetic':'300 separated beans + overlapping ellipses with >=half-visible GT, 20 repeats/set; no real-pile acceptance evidence',
            'time_gate':'Pi p95<=500ms; notebook always unverified'},card,data)
    cv2.setNumThreads(1)
    rng=np.random.default_rng(SEED)
    selected=[]
    for source in FOLDS:
        rows=sorted([r for r in data.rows if r['source'] == source and r['label'] in ('light','medium','dark')],key=lambda r:r['path'])
        selected.extend(rows[i] for i in rng.choice(len(rows),min(50,len(rows)),replace=False))
    count_dev=[r for r in data.count_dev if (data.root/r['path']).is_file()]
    # Dev ROI JPEGs already have the Agtron label removed; ordinary serving.
    chosen={r['path']:r for r in selected}
    for row in count_dev: chosen[row['path']]=row
    pred=load(args.model)
    rows=[]; count_records=[]
    for row in chosen.values():
        data.assert_safe(row)
        raw=(data.root/row['path']).read_bytes()
        from roastml.decode import decode_image
        decoded=decode_image(raw)
        raw=jpeg(np.asarray(decoded.image))  # <=1600px, uniform JPEG case
        # Original-coordinate ROI remains valid after re-encoding only if
        # original size is retained; crop beforehand using the shared helper.
        if row.get('roi'):
            from roastml.rgb_views import crop_roi
            raw=jpeg(crop_roi(decoded,row['roi']))
        metadata={'source':'agtron','roi':f'0 0 {Image.open(io.BytesIO(raw)).width} {Image.open(io.BytesIO(raw)).height}'} if row.get('roi') else {}
        for rep in range(args.repeat):
            result,stages=pred.predict_profile(raw,**metadata)
            if result['status'] in ('error','bad_image'):
                raise RuntimeError('benchmark failed on '+row['path'])
            rows.append({'path':row['path'],'set':'dev_random','rep':rep,'n':result['n_beans'],**{k:stages.get(k,0.0) for k in STAGES}})
        if row in count_dev:
            count_records.append({'path':row['path'],'scene':row['scene'],'gt':int(row['n_total']) if row.get('n_total') else None,
                 'n':result['n_beans'],'counts':{c:sum(b['label']==c for b in result['beans']) for c in ('light','medium','dark')}})
    pile,visible_gt,visible_fractions=synthetic_pile_300()
    for kind,rgb in [('synthetic_300',synthetic_300()),('synthetic_pile_300',pile)]:
        raw=jpeg(rgb)
        cv2.imwrite(str(data.root/'cache'/f'{args.out.name}_{kind}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        for rep in range(20):
            result,stages=pred.predict_profile(raw)
            rows.append({'path':kind,'set':kind,'rep':rep,'n':result['n_beans'],
                         'warnings':result['warnings'],**{k:stages.get(k,0.0) for k in STAGES}})
    latency={kind:{k:percentiles([r[k] for r in rows if r['set']==kind]) for k in STAGES} for kind in ('dev_random','synthetic_300','synthetic_pile_300')}
    counts={}
    for scene,metric,gate in [('flat','MAE',.5),('touching','MAPE',.1),('pile','MAPE',.25)]:
        labeled=[r for r in count_records if r['scene'] == scene and r['gt'] is not None]
        values=[abs((r['n'] or 0)-r['gt'])/(r['gt'] if metric == 'MAPE' else 1) for r in labeled]
        m=bootstrap(values)
        m.update(metric=metric,gate=gate,passed=m['value']<=gate if m['value'] is not None else None)
        counts[scene]=m
    sizes={name:(args.model/name).stat().st_size for name in ('model.json','bean_model.json','model_card.json') if (args.model/name).is_file()}
    summary={'latency':latency,'count_test_dev':counts,'model_size_bytes':sizes,'total_model_bytes':sum(sizes.values()),
             'memory':peak_ram(),'host':platform.platform(),'opencv_threads':cv2.getNumThreads(),
             'Pi_gate':'unverified','FW_e2e_gate':'unverified','synthetic_counts':[r['n'] for r in rows if r['set']=='synthetic_300'],
             'synthetic_pile':{'visible_gt':visible_gt,'min_visible_fraction':min(visible_fractions),
                               'predictions':[r['n'] for r in rows if r['set']=='synthetic_pile_300'],
                               'counts_are_acceptance_evidence':False},
             'count_dev_coverage':{'labeled':sum(r['gt'] is not None for r in count_records),'unlabelled':sum(r['gt'] is None for r in count_records)},
             'latency_sample_images':len(chosen),'synthetic_warning':'separated raster shapes; does not validate pile counting or roast accuracy'}
    dump(args.out/'summary.json',summary); dump(args.out/'per_image.json',rows); dump(args.out/'count_dev.json',count_records)
    # Executed notebook contains numeric outputs only; no dataset images.
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
       'cells':[{'cell_type':'markdown','metadata':{},'source':['# Per-bean notebook proxy\n','Notebook results do not pass the Raspberry Pi latency gate.\n']},
         {'cell_type':'code','execution_count':1,'metadata':{},'source':['import json\n','from pathlib import Path\n',
             "report = json.loads(Path('summary.json').read_text(encoding='utf-8'))\n", "report['latency']\n"],
          'outputs':[{'output_type':'execute_result','execution_count':1,'metadata':{},'data':{'application/json':latency,'text/plain':[json.dumps(latency,indent=2)]}}]}]}
    dump(args.out/'bench_notebook.ipynb',notebook)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
