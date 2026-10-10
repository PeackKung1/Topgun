"""Render measured A–E results; no images or models, no new experiment."""
from __future__ import annotations
import argparse
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from tools.perbean_protocol import dump

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def num(x): return 'ยังวัดไม่ได้' if x is None else f'{x:.4f}'
def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def junit(path):
    root=ET.parse(path).getroot()
    suites=list(root.findall('testsuite')) if root.tag != 'testsuite' else [root]
    return {key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}

def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('train','counter','bench','normal-tests','pi-tests','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    train=read(args.train/'summary.json'); counter=read(args.counter/'summary.json'); bench=read(args.bench/'summary.json')
    cc=counter['summary'][counter['chosen']]
    methods=('B1 broadcast','B-bean','B-bean+group')
    paired=[]; perclass=[]; pooled={}
    for fold,v in train['folds'].items():
        base=v['paired']['B1 broadcast']['image_weighted']['macro_f1']
        for name in methods:
            m=v['paired'][name]['image_weighted']
            score=m['macro_f1']
            paired.append([fold,name,num(score),num(score-base),'อ้างอิง' if name==methods[0] else ('ผ่าน' if score>=base-.02 else 'ไม่ผ่าน'),f"{v['images']-v['zero_detection_images']}/{v['images']}"])
            pc=m['f1_per_class']
            perclass.append([fold,name,*[num(pc[c]) for c in ('light','medium','dark')],num(m['light_dark_rate']),num(v['paired'][name]['unweighted']['macro_f1'])])
    for name in methods:
        cm=sum(np.array(v['paired'][name]['image_weighted']['cm']) for v in train['folds'].values())
        denom=cm[0].sum()+cm[2].sum()
        pooled[name]={'light_dark_rate':float((cm[0,2]+cm[2,0])/denom),'gate':.05,
                      'passed':bool((cm[0,2]+cm[2,0])/denom<=.05),'image_weighted_cm':cm.tolist()}
    countrows=[]
    metrics=[('flat dev',bench['count_test_dev']['flat'],.5,False),
             ('touching dev',bench['count_test_dev']['touching'],.1,False),
             ('pile dev',bench['count_test_dev']['pile'],.25,False),
             ('boos beans-mode',cc['boos_beans'],.25,False),('Empty report-half',cc['empty'],.95,True)]
    gates={}
    for name,m,gate,higher in metrics:
        value=m['value']; passed=(value>=gate if higher else value<=gate) if value is not None else None
        ci=m.get('ci95'); status='ยังวัดไม่ได้' if passed is None else ('ผ่านจุดประมาณ' if passed else 'ไม่ผ่าน')
        cross=ci is not None and (ci[0]<gate if higher else ci[1]>gate)
        if passed and cross: status+='; CI คร่อมเกณฑ์'
        countrows.append([name,m['n'],num(value),'—' if ci is None else f'{ci[0]:.4f}–{ci[1]:.4f}',('≥' if higher else '≤')+str(gate),status])
        gates[name]={'value':value,'ci95':ci,'passed_point':passed,'CI_crosses_threshold':cross}
    stage_rows=[]
    for name,stages in bench['latency'].items():
        for stage,m in stages.items(): stage_rows.append([name,stage,f"{m['p50_ms']:.3f}",f"{m['p95_ms']:.3f}"])
    tests={'normal':junit(args.normal_tests),'Pi_import_simulation':junit(args.pi_tests)}
    testpass=all(v['errors']==v['failures']==0 for v in tests.values())
    eligible=False  # Pi time and real pile/touching GT are absent regardless of notebook results.
    result={'paired_table':paired,'per_class_table':perclass,'pooled_extreme':pooled,'count_gates':gates,
            'tests':tests,'tests_passed':testpass,'deployment_eligible':eligible,
            'source_results':{'train':str(args.train.resolve()),'counter':str(args.counter.resolve()),'bench':str(args.bench.resolve())}}
    args.out.mkdir(parents=True,exist_ok=True); dump(args.out/'summary.json',result)
    parts=['<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Topgun per-bean A–E</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#19232b;background:#fafafa}table{border-collapse:collapse;width:100%;margin:20px 0;display:block;overflow:auto}th,td{border:1px solid #c6ced4;padding:8px 12px;text-align:left;white-space:nowrap}th{background:#e6edf2}h2{margin-top:36px}code{background:#eee;padding:2px 4px}p{line-height:1.7}</style>',
           '<h1>Topgun — รายงานทดลอง A–E</h1><p><strong>ยังไม่มี candidate ผ่านเกณฑ์สำหรับ deploy</strong> ผลด้านล่างใช้ trainval ที่กัน frozen ทุก group ออกแล้ว ไม่มีการเปิดภาพ manifest split=test หรือ frozen</p>',
           '<h2>Paired LOSO ราย fold</h2><p>ทุกวิธีจัดคลาสบนเมล็ดชุดเดียวกัน ถ่วง 1/n ต่อภาพ; scaler และ LR ใช้น้ำหนักเดียวกัน C เลือกด้วย inner source LOSO จาก [0.01,0.1,1,10] B1 broadcast fit ใหม่บน outer train sources ด้วย R2 Lab_hist C=.01 และ video cap10; grouping เลือก threshold ด้วยภาพปนสังเคราะห์จาก outer train fold เท่านั้น</p>',
           table(['fold','วิธี','macro-F1','Δ จาก B1','เกณฑ์ ≥B1−.02','ภาพมี detection / ทั้งหมด'],paired),
           '<p>macro-F1 เฉลี่ยสามคลาสคงที่ Safe rf_boos ไม่มี true medium หลังกัน frozen groups จึงต้องอ่าน F1 ของ medium ร่วมกับ support Zero-detection images ไม่อยู่ในคะแนนรายเมล็ด; coverage รายงานแยก ตัวนับใช้ dev config ร่วมกันทุก fold จึงเป็น nested validation ของ classifier/grouping ไม่ใช่ของทั้งระบบทั้งหมด</p>',
           '<h2>F1 รายคลาสและผิดข้ามขั้น</h2>',table(['fold','วิธี','light F1','medium F1','dark F1','light↔dark / true extremes','macro-F1 ไม่ถ่วง'],perclass),
           table(['วิธี','pooled light↔dark','เกณฑ์ ≤5%'],[[n,num(m['light_dark_rate']),'ผ่าน' if m['passed'] else 'ไม่ผ่าน'] for n,m in pooled.items()]),
           '<h2>ตัวนับเทียบ prereg</h2>',table(['ชุด/scene','n ภาพ','ค่า','95% bootstrap CI','เกณฑ์','สถานะ'],countrows),
           '<p>Empty แบ่งตาม group เป็นครึ่งจูน/ครึ่งรายงาน ครึ่งรายงานเคยเปิดในรอบก่อนแล้ว: รอบสุดท้ายใช้รายงานเท่านั้น ไม่ใช้เลือก config Bootstrap 10,000 resamples seed20261009 ต่อภาพ; เฟรมในวิดีโอสัมพันธ์กันจึงควรระวังการตีความ CI</p>',
           '<p>กฎเห็นอย่างน้อยครึ่งใช้ visible-region area ≥0.5×reference bean area เป็น heuristic ขนาดเมล็ดที่ต่างกันและการบังจริงทำให้เกณฑ์นี้คลาดเคลื่อนได้ ต้องมี GT touching/pile ที่นับตามกติกาเพื่อยืนยัน warning count_visible_only ไม่ใช่หลักฐานว่ากฎผ่าน</p>',
           '<h2>ภาพ mixed</h2>',table(['วิธี','dev class-count error','95% CI','mixed video (report only)'],[[n,num(train['mixed']['dev'][n]['value']),str(train['mixed']['dev'][n]['ci95']),num(train['mixed']['mixed_video'][n]['value'])] for n in methods]),
           table(['ภาพ dev','GT light/medium/dark','B1 error','bean error','group error','bean−B1','group−B1'],[[Path(r['path']).name,str(r['gt']),r['errors'][methods[0]],r['errors'][methods[1]],r['errors'][methods[2]],r['errors'][methods[1]]-r['errors'][methods[0]],r['errors'][methods[2]]-r['errors'][methods[0]]] for r in train['mixed']['paired_dev']]),
           '<h2>Notebook latency ราย stage (ms)</h2><p>CPU Windows, OpenCV threads1; ค่าราย stage เป็น percentile แยกกัน จึงบวก p95 เข้าด้วยกันไม่ได้ Notebook ไม่ผ่าน gate Pi p95≤500ms หรือ FW end-to-end≤1s แทนฮาร์ดแวร์จริง</p>',table(['ชุด','stage','p50','p95'],stage_rows),
           '<p>model+card '+str(bench['total_model_bytes'])+' bytes; peak RSS '+str(bench['memory'].get('peak_RSS_MiB'))+' MiB (ทั้ง process รวม input/metadata) Synthetic separated counts '+str(bench['synthetic_counts'])+'</p>',
           '<p>Overlapping synthetic pile: '+html.escape(str(bench.get('synthetic_pile')))+' เป็นการตรวจภาพสังเคราะห์ ไม่ใช่หลักฐานผ่านเกณฑ์กองจริง</p>',
           '<h2>Tests และแจ้ง FW</h2>',table(['ชุด','tests รวม','skipped','failures','errors'],[[n,m['tests'],m['skipped'],m['failures'],m['errors']] for n,m in tests.items()]),
           '<p>FW handoff: key set ของ result ไม่เพิ่ม เมื่อไม่มีเมล็ดส่ง n_beans=0, beans=[], proportions=None พร้อมผล B1 และ no_beans_detected เพิ่ม warning count_visible_only; count_method อยู่ใน profiling เท่านั้น และ estimated ตรงกับ warning bean_count_estimated การส่งข้อความยังรอผู้รับ/ช่องทางจากผู้ใช้ ไม่มีการแก้ Firmware</p>',
           '<h2>สิ่งที่ยังต้องมี</h2><p>GT touching/pile dev, การแก้ Empty ที่ยังไม่ผ่านโดยใช้ครึ่งจูนเท่านั้น, per-bean F1 ที่ผ่านทุก fold, benchmark Pi5 และ FW E2E ห้ามเปิด frozen เพื่อช่วยเลือก ไม่มีการเลือก deploy อัตโนมัติ</p>',
           '</html>']
    (args.out/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    # Windows consoles can use a non-Unicode encoding; artifact files stay UTF-8.
    print(json.dumps(result,ensure_ascii=True))

if __name__ == '__main__': main()
