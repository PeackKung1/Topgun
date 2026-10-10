"""Create a v3 preview artifact/card from measured P1 JSON, never deploy/select."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from roastml.contract import SCHEMA_VERSION, RESULT_KEYS, TIMING_KEYS
from tools.perbean_protocol import dump

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def build(card, counter, loso, bench, phase1=None):
    card = dict(card)
    for key in ('bean_counting','bean_validation','bean_run','measured','phase1_rules'):
        card.pop(key,None)
    fixed=counter['summary'][counter['chosen']]
    card.update({
        'schema_version':SCHEMA_VERSION,
        'name':card['name']+'-v3-preview',
        'contract':{
            'keys':list(RESULT_KEYS), 'timing_keys':list(TIMING_KEYS),
            'label':'count majority; tied classes resolved by probabilities; B1 fallback if n=0 or counting unavailable',
            'counts':'integer light/medium/dark; sum=n_beans; null when counting unavailable',
            'count_method':'exact or estimated; null when counting unavailable',
            'bbox':'[x,y,w,h] in oriented, resized decoded image; image_size=[w,h]',
            'empty':'n_beans=0, zero counts, beans=[], proportions=null, no_beans_detected',
            'mixed_roast':'enabled when more than one returned class has positive count',
            'timing':'ms; unavailable/non-executed named stages are zero; decode/ml/total retained'
        },
        'superseded_phase1_rules':{
            'label_override_disabled':False,'mixed_roast_disabled':False,
            'timing_drops':False,'legacy_150_cap':False,
            'scope':'flat, touching and pile; pile always estimated; physical >=half visibility is not certified'
        },
        'count_scope':{
            'visible_only':True,'occlusions_reconstructed':False,
            'visibility_min_reference_fraction':card.get('count_config',{}).get('min_visible_fraction'),
            'visibility_rule':'region area/reference area heuristic; different sizes/occlusion require real count GT',
            'pile':'experimental; counted real touching/pile GT absent; synthetic pile overcount remains'
        },
        'P1':{
            'counter':fixed,
            'paired_folds':{f:{
                'images':v['images'],'zero_detection_images':v['zero_detection_images'],
                'methods':{k:m['image_weighted'] for k,m in v['paired'].items()}
            } for f,v in loso['folds'].items()},
            'mixed_dev':loso['mixed']['dev'],
            'notebook_latency':bench['latency'],
            'count_dev':bench['count_test_dev'],
            'synthetic_pile':bench['synthetic_pile'],
            'notebook_memory':bench['memory'],
            'prior_artifact_total_bytes':bench['total_model_bytes']
        },
        'validation':'experimental preview; P1 dev gates failed/unmeasured; no new accuracy evaluation',
        'deployment_eligible':False,
        'verification':{'Pi5_p95':'unverified','FW_E2E':'separate notebook simulation only','frozen_test':'not evaluated'}
    })
    if phase1 is not None:
        # Copy only aggregate historical evidence, never its per-image results.
        card['historical_phase1']={'single_level':phase1.get('single_level'),
                                  'count':phase1.get('count'),
                                  'note':'historical protocol; superseded and not used to choose this preview'}
    return card

def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('base-model','counter','loso','bench','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase1',type=Path)
    args=p.parse_args()
    if args.out.exists():p.error('use a new preview model directory; never overwrite an existing artifact')
    sources={'counter':args.counter,'loso':args.loso,'bench':args.bench}
    if args.phase1:sources['historical_phase1']=args.phase1
    card=build(read(args.base_model/'model_card.json'),read(args.counter),read(args.loso),read(args.bench),
               read(args.phase1) if args.phase1 else None)
    card['result_sources']={name:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name,path in sources.items()}
    args.out.mkdir(parents=True)
    for name in ('model_file','bean_model_file'):
        if card.get(name):shutil.copyfile(args.base_model/card[name],args.out/card[name])
    dump(args.out/'model_card.json',card)
    print(json.dumps({'preview':str(args.out.resolve()),'schema_version':SCHEMA_VERSION,'deployment_eligible':False}))

if __name__=='__main__':main()
