"""Counts, decoded coordinate space and public timings are a single contract."""
import pytest
from roastml.api import Predictor, load
from roastml.contract import BackendOutput, LEGACY_RESULT_KEYS, RESULT_KEYS, TIMING_KEYS
from roastml.decode import decode_image
from test_contract import FakeBackend, assert_valid_result, img_bytes


def output(n=300, method='exact'):
    beans=[{'bbox':[i%20*20,i//20*20,12,8],'label':('light','medium','dark')[i%3],'conf':.8} for i in range(n)]
    return BackendOutput(probs={'light':.4,'medium':.35,'dark':.25}, n_beans=n, beans=beans,
                         proportions={'light':0,'medium':0,'dark':1}, count_method=method)


@pytest.mark.parametrize('n,method',[(0,'exact'),(300,'exact'),(300,'estimated')])
def test_v3_zero_hundreds_estimate_and_stale_proportions(n,method):
    result=Predictor(FakeBackend(lambda img:output(n,method))).predict_bytes(img_bytes((640,480)))
    assert_valid_result(result)
    assert set(LEGACY_RESULT_KEYS) < set(RESULT_KEYS)
    assert result['counts'] == dict.fromkeys(('light','medium','dark'),n//3)
    assert result['image_size'] == [640,480]
    assert result['count_method'] == method
    if n:
        assert result['label'] == 'light' and 'mixed_roast' in result['warnings']
        assert result['proportions']['dark'] < .34


def test_stub_large_oriented_image_boxes_reference_actual_decode():
    raw=img_bytes((4000,2400))
    result=load('stub',seed=5,beans_rate=1,status_weights={'ok':1}).predict_bytes(raw)
    assert_valid_result(result)
    assert result['image_size'] == list(decode_image(raw).image.size)
    assert result['image_size'] != [4000,2400]


def test_public_timing_uses_backend_profile_on_normal_request():
    class Profile(FakeBackend):
        def predict_profile(self,img,**kwargs):
            return output(0),{'WB':1.2,'segment':2.3,'count':3.4,'features':4.5,'classify':.6,'group':.7,'fallback':.8}
    result=Predictor(Profile(None)).predict_bytes(img_bytes())
    assert_valid_result(result)
    assert tuple(result['timing_ms']) == TIMING_KEYS
    assert result['timing_ms']['count'] == 3.4


@pytest.mark.parametrize('bad', ['n_mismatch','out_of_bounds','bad_conf','bad_method','not_majority'])
def test_inconsistent_backend_fails_closed(bad):
    out=output(3)
    if bad=='n_mismatch':out.n_beans=4
    if bad=='out_of_bounds':out.beans[0]['bbox']=[630,0,30,10]
    if bad=='bad_conf':out.beans[0]['conf']=float('nan')
    if bad=='bad_method':out.count_method='split'
    if bad=='not_majority':
        for bean in out.beans:bean['label']='dark'
    result=Predictor(FakeBackend(lambda img:out)).predict_bytes(img_bytes())
    assert_valid_result(result)
    assert result['status']=='error'
