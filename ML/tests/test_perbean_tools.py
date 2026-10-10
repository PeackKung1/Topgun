"""Group quarantine, equal-image weighting, and experiment immutability."""
import csv
import json
import numpy as np
import pytest
from tools.perbean_protocol import DevData, declare
from tools.train_perbean import arrays, weighted_metrics

def fixture_data(tmp_path):
    rows=[{'path':'safe.png','group':'safe','source':'ontoum224','split':'trainval','label':'light'},
          {'path':'forbidden.png','group':'frozen','source':'rf_boos','split':'trainval','label':'dark'},
          {'path':'test.png','group':'test','source':'agtron','split':'test','label':'dark'}]
    with (tmp_path/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d=tmp_path/'count_test'; d.mkdir()
    (d/'count_test_meta.json').write_text(json.dumps({'split_existing_by_group':{'frozen':'frozen'},'existing':[]}))
    (d/'count_test.csv').write_text('path,split,scene,n_total\nsafe.png,dev,flat,1\nforbidden.png,frozen,flat,1\n')
    return DevData(tmp_path)

def test_group_and_test_quarantine(tmp_path):
    data=fixture_data(tmp_path)
    assert list(data.by_path) == ['safe.png']
    for path in ('forbidden.png','test.png'):
        with pytest.raises(ValueError): data.assert_safe({'path':path})

def test_unknown_filled_web_refused(tmp_path):
    fixture_data(tmp_path)
    (tmp_path/'unknown.png').write_bytes(b'never opened')
    (tmp_path/'count_test/count_test.csv').write_text('path,split,scene,n_total\nunknown.png,dev,flat,1\n')
    with pytest.raises(ValueError,match='metadata'): DevData(tmp_path)

def test_equal_image_weights():
    records=[{'n':n,'X':np.ones((n,20)),'row':{'label':c}} for n,c in [(1,'light'),(9,'dark')]]
    X,y,w=arrays(records)
    assert w[:1].sum() == pytest.approx(1)
    assert w[1:].sum() == pytest.approx(1)
    metrics=weighted_metrics(y,y,w)
    assert metrics['image_weight_sum'] == pytest.approx(2)
    assert metrics['f1_per_class']['light'] == metrics['f1_per_class']['dark'] == 1

def test_declarations_not_overwritten(tmp_path):
    data=fixture_data(tmp_path)
    dest=tmp_path/'run'
    declare(dest,'test',{'objective':'synthetic test only'},{},data)
    first=(dest/'declaration.json').read_bytes()
    with pytest.raises(FileExistsError): declare(dest,'overwrite',{}, {},data)
    assert (dest/'declaration.json').read_bytes() == first
