"""FW regressions live in ML/tests: Firmware diffs remain uncommitted."""
import copy
import importlib
import io
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
pytest.importorskip('flask')
from roastml.api import Predictor
from roastml.contract import LEGACY_RESULT_KEYS
from test_contract import FakeBackend, img_bytes
from test_contract_v3 import output


@pytest.fixture
def fw(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'Firmware'))
    monkeypatch.setenv('TOPGUN_MQTT_ENABLED','0')
    monkeypatch.setenv('TOPGUN_DB_PATH',str(tmp_path/'test.sqlite3'))
    return importlib.import_module('service'),importlib.import_module('app')


def result(n=300,method='exact'):
    return Predictor(FakeBackend(lambda img:output(n,method))).predict_bytes(img_bytes())


def event(r):return {'msg_id':str(uuid.uuid4()),'created_at':'2026-10-10T04:00:00Z','result':r}


@pytest.mark.parametrize('n,method',[(0,'exact'),(300,'exact'),(300,'estimated')])
def test_service_full_and_compact_v3_persist_dedupe_market(fw,tmp_path,n,method):
    service,app=fw; path=tmp_path/'counts.sqlite3'
    full=result(n,method); compact=app.mqtt_result(full)
    assert 'beans' not in compact and compact['counts']==full['counts']
    for r in (full,compact):
        payload=event(r)
        assert service.persist_event(payload,path)
        assert not service.persist_event(payload,path)
    snapshot=service.read_market_snapshot(path)
    assert snapshot['bean_total']==n*2
    assert snapshot['counts']=={c:v*2 for c,v in full['counts'].items()}
    assert snapshot['estimated_results']==(2 if method=='estimated' else 0)
    assert len(snapshot['recent_results'])==2
    with service.connect_db(path) as db:
        assert [r[0] for r in db.execute('SELECT estimated FROM predictions')]==[int(method=='estimated')]*2


def test_v3_counts_win_over_conflicting_proportions(fw,tmp_path):
    service,app=fw; r=app.mqtt_result(result())
    r['proportions']={'light':0,'medium':0,'dark':1}
    service.persist_event(event(r),tmp_path/'precedence.sqlite3')
    assert service.read_market_snapshot(tmp_path/'precedence.sqlite3')['counts']==dict.fromkeys(('light','medium','dark'),100)


@pytest.mark.parametrize('bad',['sum','negative','bool','warning','method','label','unknown'])
def test_invalid_v3_rejected_without_legacy_fallback(fw,bad):
    service,app=fw; r=app.mqtt_result(result())
    if bad=='sum':r['counts']['light']=99
    if bad=='negative':r['counts']['light']=-1
    if bad=='bool':r['counts']['light']=True
    if bad=='warning':r['count_method']='estimated'
    if bad=='method':r['count_method']='split'
    if bad=='label':r['counts']={'light':0,'medium':0,'dark':300}
    if bad=='unknown':r['extra']=1
    with pytest.raises(ValueError):service.validate_event(event(r))


def test_legacy_v2_aggregate_fallback_and_existing_schema_migration(fw,tmp_path):
    service,app=fw; path=tmp_path/'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE predictions (msg_id TEXT PRIMARY KEY,created_at TEXT NOT NULL,status TEXT NOT NULL,label TEXT,confidence REAL,model TEXT NOT NULL,result_json TEXT NOT NULL)')
    r={k:v for k,v in result().items() if k in LEGACY_RESULT_KEYS}
    r['proportions']=dict.fromkeys(('light','medium','dark'),1/3)
    r['beans']=[]
    service.persist_event(event(r),path)
    assert service.read_market_snapshot(path)['counts']==dict.fromkeys(('light','medium','dark'),100)
    assert app.mqtt_result(r)['beans']==[]


def test_null_v3_counting_does_not_reconstruct_legacy_counts(fw,tmp_path):
    service,app=fw; r=app.mqtt_result(result())
    r.update(n_beans=None,counts=None,count_method=None)
    service.persist_event(event(r),tmp_path/'null.sqlite3')
    assert service.read_market_snapshot(tmp_path/'null.sqlite3')['bean_total']==0


def test_http_keeps_beans_but_queues_only_aggregate(fw,monkeypatch,tmp_path):
    service,app=fw; queued=[]
    monkeypatch.setattr(app,'enqueue_prediction',lambda payload:queued.append(payload))
    predictor=Predictor(FakeBackend(lambda img:output(300,'estimated')))
    client=app.create_app(predictor,start_background=False).test_client()
    response=client.post('/api/predict',data={'image':(io.BytesIO(img_bytes()),'beans.jpg')})
    r=response.get_json()
    assert response.status_code==200 and len(r['beans'])==300
    assert len(queued)==1 and 'beans' not in queued[0]
    assert queued[0]['counts']==r['counts'] and queued[0]['warnings']==r['warnings']
    assert len(json.dumps(queued[0])) < len(json.dumps(r))/4
    assert service.persist_event(event(queued[0]),tmp_path/'http.sqlite3')
    for route in ('/','/market','/api/market'):
        assert client.get(route).status_code==200
    assert client.post('/api/predict').status_code==400
    assert client.post('/api/predict',data={'image':(io.BytesIO(b''),'empty.jpg')}).status_code==400


def test_mqtt_queue_failure_does_not_fail_http(fw,monkeypatch):
    _,app=fw
    def broken(_):raise RuntimeError('offline queue')
    monkeypatch.setattr(app,'enqueue_prediction',broken)
    client=app.create_app(Predictor(FakeBackend(lambda img:output(0))),start_background=False).test_client()
    assert client.post('/api/predict',data={'image':(io.BytesIO(img_bytes()),'empty.jpg')}).status_code==200


def test_concurrent_first_database_migration_and_dedupe(fw,tmp_path):
    service,app=fw; path=tmp_path/'concurrent.sqlite3'
    payload=event(app.mqtt_result(result(300,'estimated')))
    with ThreadPoolExecutor(max_workers=4) as pool:
        inserted=list(pool.map(lambda _:service.persist_event(payload,path),range(4)))
    assert sum(inserted)==1
    assert service.read_market_snapshot(path)['bean_total']==300
