"""Per-bean contract, shared features, grouping and failure paths on synthetic images."""
from __future__ import annotations
import io
import os
import subprocess
import sys
from collections import Counter
import cv2
import numpy as np
import pytest
from PIL import Image
from test_contract import assert_valid_result
from test_linear_backend import write_model
from roastml.api import ModelLoadError, load
from roastml.counter import CountConfig, count_beans
from roastml.features import FEATURES_ALL, pixel_stats, pixel_stats_grouped
from roastml.perbean import bean_features, count_features, group_predictions, majority_probs
from roastml.segment import SegConfig, prepare_image, segment

LIGHT, MEDIUM, DARK = (170,125,85), (110,72,45), (50,32,22)

def rgb_scene(n=6, touching=False, size=(640,480)):
    w,h=size
    rgb=np.full((h,w,3),235,np.uint8)
    for i in range(n):
        x=120+(i%4)*(49 if touching else 125)
        y=130+(i//4)*125
        cv2.ellipse(rgb,(x,y),(29,20),0,0,360,(LIGHT,MEDIUM,DARK)[i%3],-1)
    return rgb

def encoded(rgb):
    b=io.BytesIO(); Image.fromarray(rgb).save(b,'PNG'); return b.getvalue()

@pytest.mark.parametrize('n',[0,1,5,12])
def test_synthetic_counts_and_invariants(tmp_path,n):
    pred=load(write_model(tmp_path/'m','b1_linear_beans'))
    r=pred.predict_bytes(encoded(rgb_scene(n)))
    assert_valid_result(r)
    assert r['n_beans'] == n == len(r['beans'])
    assert sum(Counter(b['label'] for b in r['beans']).values()) == n
    if n:
        counts=Counter(b['label'] for b in r['beans'])
        assert counts[r['label']] == max(counts.values())
        for c in ('light','medium','dark'):
            assert r['proportions'][c] == pytest.approx(counts[c]/n,abs=1e-4)
    else:
        assert r['proportions'] is None and 'no_beans_detected' in r['warnings']
        baseline=load(write_model(tmp_path/'b1'))
        rb=baseline.predict_bytes(encoded(rgb_scene(0)))
        assert r['probs'] == rb['probs'] and r['label'] == rb['label']

@pytest.mark.parametrize('touching',[False,True])
def test_count_touching(touching):
    result=count_beans(rgb_scene(3,touching=touching))
    assert result.n == 3
    assert set(np.unique(result.labels)) == {0,1,2,3}
    assert all(np.any(result.core & (result.labels == i)) for i in range(1,4))
    assert len(result.bboxes) == result.n

@pytest.mark.parametrize('seed',[0,9,12])
def test_grouped_feature_parity(seed):
    rng=np.random.default_rng(seed)
    labels=np.repeat(np.arange(31),rng.integers(1,200,size=31))
    rng.shuffle(labels)
    lab=rng.normal((45,4,10),(32,26,30),(len(labels),3))
    expected=np.stack([pixel_stats(lab[labels == i]) for i in range(31)])
    np.testing.assert_allclose(pixel_stats_grouped(lab,labels,31),expected,atol=2e-12,rtol=1e-12)

@pytest.mark.parametrize('labels,n',[([-1,0],1),([0,2],2),([0,0],2)])
def test_invalid_grouped_features(labels,n):
    with pytest.raises(ValueError): pixel_stats_grouped(np.ones((2,3)),labels,n)

def test_dust_cannot_define_reference_size():
    rgb=rgb_scene(1)
    for x,y in [(50,50),(590,50),(590,410)]:
        cv2.ellipse(rgb,(x,y),(5,4),0,0,360,MEDIUM,-1)
    assert count_beans(rgb).n == 1

def test_groove_is_not_two_beans():
    rgb=rgb_scene(1)
    cv2.line(rgb,(120,116),(120,144),(50,35,24),3)
    assert count_beans(rgb).n == 1

def test_area_option_is_deterministic_across_threads():
    from concurrent.futures import ThreadPoolExecutor
    from roastml.counter import _split_area
    mask=np.zeros((70,130),np.uint8)
    cv2.ellipse(mask,(40,35),(30,23),0,0,360,1,-1)
    cv2.ellipse(mask,(90,35),(30,23),0,0,360,1,-1)
    cfg=CountConfig(touch_method='area')
    def run(_): return _split_area(mask.astype(bool),2,cfg)
    with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,range(8)))
    for result in results: np.testing.assert_array_equal(result,results[0])
    assert results[0].max() == 2

def test_sub_half_reference_fragment_excluded():
    rgb=rgb_scene(3)
    cv2.ellipse(rgb,(500,350),(29,20),0,0,360,MEDIUM,-1)
    rgb[330:371,480:507]=235
    assert count_beans(rgb).n == 3

def test_half_reference_config_cannot_be_lowered():
    with pytest.raises(ValueError): CountConfig.from_dict({'min_visible_fraction':.4})

def test_overlapping_synthetic_truth_and_pile_warning(tmp_path):
    from tools.bench_perbean import synthetic_pile_300
    rgb,n,fractions=synthetic_pile_300()
    assert n == 300 and min(fractions) >= .5
    pred=load(write_model(tmp_path/'m','b1_linear_beans'))
    result,stages=pred.predict_profile(encoded(rgb))
    assert_valid_result(result)
    assert result['n_beans'] == len(result['beans'])
    if result['n_beans']:
        assert 'count_visible_only' in result['warnings']
        assert (stages['count_method'] == 'estimated') == ('bean_count_estimated' in result['warnings'])

def test_neutral_metal_rejected():
    rgb=np.full((300,400,3),80,np.uint8)
    cv2.line(rgb,(15,20),(370,270),(20,20,20),8)
    cv2.circle(rgb,(200,140),55,(45,45,45),-1)
    assert count_beans(rgb,CountConfig(metal_chroma_min=16)).n == 0

@pytest.mark.parametrize('blue_factor',[1.0,.92,.8])
def test_train_and_serve_features_wb_parity(tmp_path,monkeypatch,blue_factor):
    rgb=rgb_scene(6)
    rgb[...,2]=(rgb[...,2]*blue_factor).astype(np.uint8)
    sc,cc=SegConfig(),CountConfig()
    prep=prepare_image(rgb,sc)
    a,X=bean_features(rgb,cc,sc)
    b,Y=bean_features(rgb,cc,sc,prepared=prep)
    np.testing.assert_array_equal(a.labels,b.labels)
    np.testing.assert_array_equal(X,Y)
    assert a.wb_applied == b.wb_applied == (blue_factor >= .9)
    np.testing.assert_array_equal(segment(rgb,sc).lab,segment(rgb,sc,prepared=prep).lab)
    pred=load(write_model(tmp_path/'m','b1_linear_beans'),warmup=False)
    captured=[]
    original=pred.backend.bean_model.proba
    def capture(v):
        captured.append(v.copy()); return original(v)
    monkeypatch.setattr(pred.backend.bean_model,'proba',capture)
    r=pred.predict_bytes(encoded(rgb)); assert_valid_result(r)
    np.testing.assert_array_equal(captured[-1],X)

def test_bbox_coordinates_after_decode(tmp_path):
    rgb=np.full((1500,2000,3),235,np.uint8)
    cv2.ellipse(rgb,(1000,750),(90,60),0,0,360,DARK,-1)
    r=load(write_model(tmp_path/'m','b1_linear_beans')).predict_bytes(encoded(rgb))
    assert_valid_result(r)
    x,y,w,h=r['beans'][0]['bbox']
    assert abs(x+w/2-800)<5 and abs(y+h/2-600)<5

def test_group_uses_mean_probability_and_order():
    X=np.zeros((9,len(FEATURES_ALL))); X[:,0]=np.repeat([20,42,70],3)
    P=np.tile([.2,.3,.5],(9,1))
    ids,conf,info=group_predictions(X,P,['light','medium','dark'],{'min_L_gap':12,'max_k':3})
    assert info['k'] == 3
    assert ids.tolist() == [2]*3+[1]*3+[0]*3
    assert np.all((conf>=0)&(conf<=1))

def test_majority_formula():
    ids=np.array([0,0,0,1,2])
    P=np.array([[.5,.4,.1]]*3+[[.1,.8,.1],[.1,.1,.8]])
    p,prop=majority_probs(ids,P,['light','medium','dark'])
    np.testing.assert_allclose(list(p.values()),.999*np.array([.6,.2,.2])+.001*P.mean(0))
    assert max(p,key=p.get) == 'light'

def test_grouped_majority_one_vote_margin_at_1000():
    X=np.zeros((1000,len(FEATURES_ALL)))
    X[:,0]=np.repeat([20,42,70],[334,333,333])
    P=np.tile([1.0,0.0,0.0],(1000,1))
    labels,_,_=group_predictions(X,P,['light','medium','dark'],{'min_L_gap':12,'max_k':3})
    probs,_=majority_probs(labels,P,['light','medium','dark'])
    counts=np.bincount(labels,minlength=3)
    assert counts[['light','medium','dark'].index(max(probs,key=probs.get))] == counts.max()

def test_300_beans_no_150_cap(tmp_path):
    rgb=np.full((600,900,3),235,np.uint8)
    for i in range(300):
        cv2.ellipse(rgb,(20+(i%20)*44,20+(i//20)*39),(13,9),0,0,360,MEDIUM,-1)
    r=load(write_model(tmp_path/'m','b1_linear_beans')).predict_bytes(encoded(rgb))
    assert_valid_result(r)
    assert r['n_beans'] == len(r['beans']) == 300

def test_off_switch_and_legacy_timeout_do_not_drop_counts(tmp_path,monkeypatch):
    rgb=encoded(rgb_scene(5))
    old=load(write_model(tmp_path/'old','b1_linear_beans',bean_counting={'budget_ms':1e-6,'max_blobs':3}))
    assert old.predict_bytes(rgb)['n_beans'] == 5
    monkeypatch.setenv('ROAST_BEANS','0')
    off=load(write_model(tmp_path/'off','b1_linear_beans'))
    b1=load(write_model(tmp_path/'b1'))
    a,b=off.predict_bytes(rgb),b1.predict_bytes(rgb)
    for k in ('label','probs','warnings','n_beans','beans','proportions'): assert a[k] == b[k]

def test_count_failure_returns_b1(tmp_path,monkeypatch):
    import roastml.bean_backend as bb
    def fail(*a,**k): raise RuntimeError('synthetic failure')
    monkeypatch.setattr(bb,'count_beans',fail)
    pred=load(write_model(tmp_path/'m','b1_linear_beans'),warmup=False)
    r=pred.predict_bytes(encoded(rgb_scene(2)))
    assert_valid_result(r)
    assert r['n_beans'] is None and 'no_beans_detected' in r['warnings']

@pytest.mark.parametrize('raw',[b'',b'%PDF-1.4\n%%EOF',None,os.urandom(4000)],ids=['empty','pdf','null','noise'])
def test_bad_input(tmp_path,raw):
    assert_valid_result(load(write_model(tmp_path/'m','b1_linear_beans')).predict_bytes(raw))

def test_tiny_image(tmp_path):
    pred=load(write_model(tmp_path/'m','b1_linear_beans'))
    assert_valid_result(pred.predict_bytes(encoded(np.full((1,1,3),DARK,np.uint8))))

def test_no_training_runtime_dependencies(tmp_path):
    d=write_model(tmp_path/'m','b1_linear_beans')
    code="import sys,roastml.api as a; a.load(sys.argv[1]); bad=[x for x in ('sklearn','torch','onnx','scipy','matplotlib','pandas') if x in sys.modules]; print(bad); sys.exit(bool(bad))"
    r=subprocess.run([sys.executable,'-c',code,str(d)],capture_output=True,text=True,env={**os.environ,'PYTHONPATH':os.pathsep.join(sys.path)})
    assert r.returncode == 0,r.stdout+r.stderr

@pytest.mark.parametrize('config',[{'max_beans':0},{'touch_method':'edge'},{'core_frac':float('nan')},{'bad':1}])
def test_invalid_count_config(tmp_path,config):
    with pytest.raises(ModelLoadError): load(write_model(tmp_path/'m','b1_linear_beans',count_config=config))
