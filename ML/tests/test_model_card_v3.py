"""Updated card values come from measured files; no per-image evidence leaks."""
import json
from tools.model_card_v3 import build


def test_generated_card_supersedes_phase1_but_retains_measured_values():
    old={'name':'fixture','count_config':{'min_visible_fraction':.5},'bean_counting':{'max_blobs':150},
         'bean_validation':'no mixed_roast, no label override','measured':{'stale':True}}
    counter={'chosen':'fixed','summary':{'fixed':{'empty':{'value':.91}}}}
    loso={'folds':{'source':{'images':7,'zero_detection_images':1,'paired':{'B-bean':{'image_weighted':{'macro_f1':.37}}}}},'mixed':{'dev':{'value':2}}}
    bench={'latency':{'dev':{'total':{'p95_ms':123}}},'count_test_dev':{},'synthetic_pile':{'visible_gt':300,'predictions':[514]},'memory':{},'total_model_bytes':900}
    phase1={'single_level':{'ALL':{'label_flip_rate':0}},'count':{'MAE':.2},'per_image':{'private':'not copied'}}
    updated=build(old,counter,loso,bench,phase1)
    assert old['bean_counting']['max_blobs']==150
    assert 'bean_counting' not in updated and 'measured' not in updated
    assert updated['P1']['counter']==counter['summary']['fixed']
    assert updated['P1']['paired_folds']['source']['methods']['B-bean']['macro_f1']==.37
    assert updated['P1']['synthetic_pile']==bench['synthetic_pile']
    assert updated['historical_phase1']['count']==phase1['count']
    assert 'per_image' not in json.dumps(updated)
    assert updated['schema_version']==3 and not updated['deployment_eligible']
