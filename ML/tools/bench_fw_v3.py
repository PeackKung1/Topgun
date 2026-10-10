"""Local HTTP shaping + actual FW browser submit-to-canvas benchmark.
Only synthetic inputs; bind loopback; MQTT broker disabled (queue still exercised).
Open /__bench on LAN/hotspot proxy ports and click Start: 20 timed requests each.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tools.bench_perbean import synthetic_300, synthetic_pile_300
from tools.perbean_protocol import DevData, SEED, declare, dump
from roastml.paths import data_dir

PROFILES={'LAN':{'rtt_ms':2.0,'up_mbps':100.0,'down_mbps':100.0},
          'hotspot':{'rtt_ms':20.0,'up_mbps':20.0,'down_mbps':50.0}}
SCENES=['empty','separated300','separated300','pile300','separated300']*4
# Uses the production form handler, compression, HTTP endpoint, summary and canvas.
HARNESS=r'''<!doctype html><html lang="th"><meta charset="utf-8">
<title>Topgun notebook E2E</title>
<style>body{font:15px system-ui;margin:20px;background:#f4f3ed}button{padding:10px 20px}iframe{display:block;width:min(680px,100%);height:calc(100vh - 240px);min-height:400px;border:1px solid #ccc}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>
<h1>Notebook E2E · __PROFILE__</h1><p>เครือข่ายจำลอง __CONFIG__ · MQTT broker ปิด · ใช้ภาพสังเคราะห์</p>
<button id="start">เริ่มวัด 20 ครั้ง</button><p id="progress" role="status">พร้อม</p>
<iframe id="app" src="/" title="หน้าอัปโหลดจริง"></iframe>
<table><thead><tr><th>รอบ</th><th>ภาพ</th><th>จำนวน</th><th>E2E ms</th><th>วาด ms</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const frame=document.querySelector('#app'), start=document.querySelector('#start'), statusText=document.querySelector('#progress');
const scenes=__SCENES__;
const profile=__PROFILE_JSON__;
let busy=false;
async function submit(scene) {
  const response=await fetch('/__bench/fixture/'+scene+'.jpg');
  const blob=await response.blob();
  const doc=frame.contentDocument, transfer=new DataTransfer();
  transfer.items.add(new File([blob],scene+'.jpg',{type:'image/jpeg'}));
  const input=doc.querySelector('#image-input');
  input.files=transfer.files;
  input.dispatchEvent(new Event('change',{bubbles:true}));
  doc.querySelector('#preview-wrap').scrollIntoView({block:'center'});
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{cleanup();reject(new Error('client benchmark timeout'));},30000);
    const cleanup=()=>{clearTimeout(timer);doc.removeEventListener('topgun:prediction-rendered',done);doc.removeEventListener('topgun:prediction-error',fail);};
    const done=e=>{cleanup();resolve(e.detail);};
    const fail=e=>{cleanup();reject(new Error(e.detail.message));};
    doc.addEventListener('topgun:prediction-rendered',done);
    doc.addEventListener('topgun:prediction-error',fail);
    doc.querySelector('#predict-form').requestSubmit();
  });
}
start.addEventListener('click',async()=>{
 if(busy)return;busy=true;start.disabled=true;
 try {
  statusText.textContent='อุ่นเครื่อง 2 ครั้ง (ไม่นับใน 20)';
  await submit('empty');await submit('separated300');
  for(let i=0;i<scenes.length;i++){
   statusText.textContent='กำลังวัด '+(i+1)+'/20';
   const detail=await submit(scenes[i]);
   const canvas=frame.contentDocument.querySelector('#preview').getBoundingClientRect();
   const row={profile,iteration:i+1,scene:scenes[i],...detail,user_agent:navigator.userAgent,
    visibility_state:document.visibilityState,canvas_rect:{top:canvas.top,bottom:canvas.bottom},
    frame_rect:{top:frame.getBoundingClientRect().top,bottom:frame.getBoundingClientRect().bottom},
    viewport_height:innerHeight,frame_height:frame.clientHeight};
   const response=await fetch('/__bench/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)});
   if(!response.ok)throw new Error(await response.text());
   const tr=document.createElement('tr');
   for(const value of [i+1,scenes[i],detail.n_beans,detail.e2e_ms.toFixed(1),detail.draw_ms.toFixed(1)]){
    const td=document.createElement('td');td.textContent=value;tr.append(td);
   }
   document.querySelector('#rows').append(tr);
  }
  statusText.textContent='เสร็จ 20/20 · บันทึกผลแล้ว';
 }catch(error){statusText.textContent='ERROR: '+error.message;}
});
</script></html>'''

def pct(values):
    return {'n':len(values),'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95))} if values else {'n':0,'p50_ms':None,'p95_ms':None}

class Records:
    def __init__(self,out):
        self.out=out;self.rows=[];self.lock=threading.Lock()
    def append(self,row):
        if row.get('profile') not in PROFILES or not isinstance(row.get('iteration'),int) or not 1 <= row['iteration'] <= 20:
            raise ValueError('invalid benchmark profile/iteration')
        if row.get('scene') != SCENES[row['iteration']-1] or row.get('schema_version') != 3 or row.get('status') not in ('ok','low_confidence'):
            raise ValueError('unexpected scene/schema/status')
        if any(not isinstance(row.get(k),(int,float)) or not np.isfinite(row[k]) or row[k]<0 for k in ('e2e_ms','draw_ms','upload_bytes')):
            raise ValueError('invalid browser measurement')
        with self.lock:
            if any((r['profile'],r['iteration'])==(row['profile'],row['iteration']) for r in self.rows):
                raise ValueError('duplicate measurement; start a new declared run')
            self.rows.append(row)
            dump(self.out/'browser_measurements.json',self.rows)
            summary={'profiles':{},'measurement':'production browser performance.now from submit through two animation frames after canvas draw',
                     'network':'loopback HTTP proxy shapes request upload and JSON download plus declared RTT; no radio/LAN hardware',
                     'MQTT':'aggregate enqueue exercised; broker disabled','host':platform.platform(),
                     'accuracy':'synthetic fixtures only; no model accuracy gate or Pi measurement'}
            for profile in PROFILES:
                rr=[r for r in self.rows if r['profile']==profile]
                summary['profiles'][profile]={'network':PROFILES[profile],'E2E':pct([r['e2e_ms'] for r in rr]),
                    'draw':pct([r['draw_ms'] for r in rr]),'ML':pct([r['timing_ms']['ml'] for r in rr]),
                    'by_scene':{s:{'E2E':pct([r['e2e_ms'] for r in rr if r['scene']==s]),
                                   'predicted_counts':[r['n_beans'] for r in rr if r['scene']==s]} for s in ('empty','separated300','pile300')},
                    'complete':len(rr)==20,'p95_under_1s':pct([r['e2e_ms'] for r in rr])['p95_ms']<1000 if rr else None}
            dump(self.out/'summary.json',summary)
            with (self.out/'measurements.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['profile','iteration','scene','e2e_ms','draw_ms','upload_bytes','n_beans','ml_ms'])
                writer.writeheader()
                for r in self.rows:writer.writerow({**{k:r[k] for k in writer.fieldnames if k!='ml_ms'},'ml_ms':r['timing_ms']['ml']})

def handler(profile,origin_port,fixtures,records):
    cfg=PROFILES[profile]
    class Proxy(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def reply(self,body,kind='text/html; charset=utf-8',status=200,shaped=False):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers()
            start=time.perf_counter()
            for pos in range(0,len(body),16384):
                chunk=body[pos:pos+16384]
                if shaped:time.sleep(max(0,(pos+len(chunk))*8/(cfg['down_mbps']*1e6)-(time.perf_counter()-start)))
                self.wfile.write(chunk)
            self.wfile.flush()
        def read_body(self,shaped=False):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<=length<=33*1024*1024:raise ValueError('request too large')
            chunks=[];read=0;start=time.perf_counter()
            while read<length:
                chunk=self.rfile.read(min(16384,length-read))
                if not chunk:raise ValueError('truncated upload')
                read+=len(chunk);chunks.append(chunk)
                if shaped:time.sleep(max(0,read*8/(cfg['up_mbps']*1e6)-(time.perf_counter()-start)))
            return b''.join(chunks)
        def do_GET(self):
            if self.path=='/__bench':
                page=HARNESS.replace('__PROFILE__',profile).replace('__CONFIG__',json.dumps(cfg)).replace(
                    '__PROFILE_JSON__',json.dumps(profile)).replace('__SCENES__',json.dumps(SCENES))
                return self.reply(page.encode('utf-8'))
            if self.path.startswith('/__bench/fixture/'):
                scene=self.path.rsplit('/',1)[-1].removesuffix('.jpg')
                return self.reply(fixtures[scene],'image/jpeg') if scene in fixtures else self.reply(b'not found',status=404)
            self.forward()
        def do_POST(self):
            if self.path=='/__bench/record':
                try:
                    row=json.loads(self.read_body())
                    if row.get('profile')!=profile:raise ValueError('wrong proxy profile')
                    records.append(row)
                    return self.reply(b'{"saved":true}','application/json')
                except (ValueError,KeyError) as e:return self.reply(str(e).encode(),status=400)
            self.forward()
        def forward(self):
            shaped=self.path=='/api/predict'
            body=self.read_body(shaped) if self.command=='POST' else None
            if shaped:time.sleep(cfg['rtt_ms']/1000)
            conn=http.client.HTTPConnection('127.0.0.1',origin_port,timeout=30)
            try:
                headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','connection','content-length')}
                conn.request(self.command,self.path,body=body,headers=headers)
                response=conn.getresponse();content=response.read()
                self.reply(content,response.getheader('Content-Type','application/octet-stream'),response.status,shaped)
            finally:conn.close()
    return Proxy

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=8810)
    args=p.parse_args()
    data=DevData(data_dir());card=json.loads((args.model/'model_card.json').read_text(encoding='utf-8'))
    fw=Path(__file__).resolve().parents[2]/'Firmware'
    hashes={str(f.relative_to(fw)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [fw/'app.py',fw/'service.py',fw/'static/script.js',fw/'static/style.css',fw/'templates/index.html',fw/'templates/market.html']}
    declare(args.out,'contract-v3-notebook-browser-e2e',{
        'n':'20 timed runs/profile; two explicit unmeasured warmups/profile',
        'selection':'fixed fixtures/order; no model/count/group tuning',
        'fixtures':SCENES,'endpoint':'performance.now from production submit handler through canvas paint; not fetch-only',
        'paint_context':'visible browser; iframe before result table; preview scrolled into frame viewport before submit; report offscreen pilot separately',
        'network':PROFILES,'gate':'notebook report p95<1000ms; cannot establish Pi5 or real network gate',
        'MQTT':'broker disabled; existing queue remains in HTTP path'},{
        'card_sha256':hashlib.sha256((args.model/'model_card.json').read_bytes()).hexdigest(),
        'FW_source_sha256':hashes,'model':str(args.model.resolve()),'seed':SEED},data)
    cv2.setNumThreads(1)
    arrays={'empty':np.full((600,900,3),235,np.uint8),'separated300':synthetic_300(),'pile300':synthetic_pile_300()[0]}
    fixtures={}
    for name,rgb in arrays.items():
        buf=io.BytesIO();Image.fromarray(rgb).save(buf,'JPEG',quality=85);fixtures[name]=buf.getvalue()
        (data.root/'cache'/f'{args.out.name}_{name}.jpg').write_bytes(fixtures[name])
    os.environ['TOPGUN_MQTT_ENABLED']='0';os.environ['ROASTML_MODEL']=str(args.model.resolve())
    os.environ['TOPGUN_DB_PATH']=str((args.out/'benchmark.sqlite3').resolve())
    sys.path.insert(0,str(fw))
    import app as firmware_app
    if firmware_app.app.extensions['roastml_load_error']:raise RuntimeError('benchmark model failed to load')
    from waitress import create_server
    origin=create_server(firmware_app.app,host='127.0.0.1',port=args.port,threads=2)
    threading.Thread(target=origin.run,daemon=True).start()
    records=Records(args.out);servers=[]
    for i,profile in enumerate(PROFILES,1):
        proxy=ThreadingHTTPServer(('127.0.0.1',args.port+i),handler(profile,args.port,fixtures,records))
        threading.Thread(target=proxy.serve_forever,daemon=True).start();servers.append(proxy)
        print(json.dumps({'profile':profile,'url':f'http://127.0.0.1:{args.port+i}/__bench'}),flush=True)
    try:
        while True:threading.Event().wait(1)
    except KeyboardInterrupt:
        for server in servers:server.shutdown();server.server_close()
        origin.close()

if __name__=='__main__':main()
