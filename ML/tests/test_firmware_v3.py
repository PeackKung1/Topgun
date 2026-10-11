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


def test_market_snapshot_incremental_matches_full_rebuild(fw,tmp_path):
    """read_market_snapshot folds in only new rows: after every insert it must equal a from-scratch rebuild."""
    service,app=fw; path=tmp_path/'incremental.sqlite3'
    def rebuilt(limit):
        kept=dict(service._SNAPSHOTS); service._SNAPSHOTS.clear()
        try:return service.read_market_snapshot(path,limit)
        finally:service._SNAPSHOTS.clear(); service._SNAPSHOTS.update(kept)
    def check():
        for limit in (50,3):
            assert service.read_market_snapshot(path,limit)==rebuilt(limit)
    failed=app.mqtt_result(result(3)); failed.update(status='bad_image',label=None,n_beans=None,counts=None,count_method=None,warnings=[])
    legacy={k:v for k,v in result(6).items() if k in LEGACY_RESULT_KEYS}   # v2: beans listed, no counts key
    stamps=['2026-10-10T04:00:05Z','2026-10-10T04:00:01Z','2026-10-10T04:00:09Z','2026-10-10T04:00:01Z','2026-10-10T04:00:07Z',
            '2026-10-10T04:00:03Z','2026-10-10T04:00:08Z','2026-10-10T04:00:02Z']   # arrive out of order, with a tie
    payloads=[result(3),app.mqtt_result(result(30,'estimated')),failed,legacy,app.mqtt_result(result(0)),result(60),
              app.mqtt_result(result(9)),legacy]
    assert service.read_market_snapshot(path)['bean_total']==0
    for stamp,r in zip(stamps,payloads):
        assert service.persist_event({**event(copy.deepcopy(r)),'created_at':stamp},path)
        check()
    snapshot=service.read_market_snapshot(path,3)
    assert snapshot['bean_total']==3+30+6+0+60+9+6 and snapshot['estimated_results']==1
    assert [r['created_at'][-3:-1] for r in snapshot['recent_results']]==['08','07','05']        # failed 09 is skipped
    assert [(b['created_at'][-3:-1],b['index']) for b in snapshot['recent_beans']]==[('05',1),('05',2),('05',3)]
    assert service.read_market_snapshot(path,3) is not snapshot and service.read_market_snapshot(path,3)==snapshot
    snapshot['counts']['light']=-1; snapshot['recent_results'][0]['counts']['light']=-1          # callers cannot corrupt the cache
    check()
    with service.connect_db(path) as db:                                                         # history trimmed by hand → rebuild
        db.execute("DELETE FROM predictions WHERE created_at='2026-10-10T04:00:07Z'")
    check()
    assert service.read_market_snapshot(path)['bean_total']==3+30+6+60+9+6
    assert service.persist_event({**event(result(3)),'created_at':'2026-10-10T04:00:10Z'},path)  # delete + insert between two reads
    with service.connect_db(path) as db:
        db.execute("DELETE FROM predictions WHERE created_at='2026-10-10T04:00:05Z'")
    check()
    path.unlink(); [p.unlink() for p in tmp_path.glob('incremental.sqlite3-*')]                   # history wiped while running
    assert service.read_market_snapshot(path)['bean_total']==0
    assert service.persist_event(event(result(3)),path) and service.read_market_snapshot(path)['bean_total']==3


def test_market_snapshot_parses_only_new_rows(fw,tmp_path,monkeypatch):
    service,app=fw; path=tmp_path/'calls.sqlite3'; parsed=[]
    real=service._row_contribution
    monkeypatch.setattr(service,'_row_contribution',lambda row:parsed.append(row['rowid']) or real(row))
    for _ in range(5):service.persist_event(event(app.mqtt_result(result(3))),path)
    assert service.read_market_snapshot(path)['bean_total']==15 and len(parsed)==5
    assert service.read_market_snapshot(path)['bean_total']==15 and len(parsed)==5               # nothing new → nothing parsed
    service.persist_event(event(app.mqtt_result(result(3))),path)
    assert service.read_market_snapshot(path)['bean_total']==18 and len(parsed)==6               # one new row → one parse
