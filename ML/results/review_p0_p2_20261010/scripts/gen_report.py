"""Render review report (single HTML) + summary.json from the evidence JSON files produced by run_e.py / d3.py / c23.py."""
import json, sys, html
from pathlib import Path

out = Path(sys.argv[1])
E = json.load(open(out / "evidence_contract.json", encoding="utf-8"))
D = json.load(open(out / "evidence_perbean.json", encoding="utf-8"))
C = json.load(open(out / "evidence_counter.json", encoding="utf-8"))
esc = html.escape

VERDICTS = [
 ("A1", "PASS", "main = origin/main (ref ในเครื่อง) = 69c4354 ไม่ ahead/behind · Firmware/ มี 7 commit ทั้งหมดโดย Kittiphob Subsorn · c02ed6a/3a7172d/d3a6dcb แตะไฟล์นอก ML/ = 0 ไฟล์ · (ไม่ได้ fetch → ความสดของ remote = UNVERIFIED · worktree seed-count มี Firmware แก้ค้าง 6 ไฟล์ ไม่ได้ commit)"),
 ("A2", "UNVERIFIED", "ทำซ้ำไม่ได้: `git -C <main> status --porcelain` และ `git diff --stat` ว่างทั้งคู่ ณ ตอนตรวจ · ไม่มี .gitattributes · ls-files --eol = 241 ไฟล์ i/lf w/crlf · อ่าน core.autocrlf ไม่ได้ (คำสั่งถูกปฏิเสธสิทธิ์)"),
 ("A3", "FAIL", "ค่า default โหลด models/current = backend b1_linear (สร้าง 8 ต.ค.) → ไม่เรียก counter.py เลย · n_beans/counts/count_method = null · beans = [] · ROAST_BEANS ไม่มีผล · label มาจาก softmax ระดับภาพ"),
 ("A4", "PASS", "card ของ current: LOSO 0.6382 / 0.6774 ตรง results/baseline_loso.json:136,713 · เป็น card ของ B1 จึงไม่มีเกณฑ์รายเมล็ด · card preview (ไม่ได้ deploy) ตรงไฟล์ P1 และเขียน 'P1 dev gates failed/unmeasured', deployment_eligible=false"),
 ("B1", "FAIL", "prereg ไม่ถูกแก้หลังเห็นผล (blob 48431eb commit เดียว c02ed6a · mtime 9 ต.ค. 22:57 < ไฟล์ผล P1 ตัวแรก 10 ต.ค. 00:33) และเกณฑ์ 8 ตัวตรง GATES · แต่ status ยัง 'DRAFT … no P1 experiment has been run' = รัน P1 บน prereg ที่ไม่เคยอนุมัติ + เบี่ยงจาก prereg 3 จุด"),
 ("B2", "FAIL", "40 แถว · กรอกแล้ว 11 (dev 3 + frozen 8) ทั้งหมดเป็น rf_boos ที่ prefill จาก YOLO · คนนับจริง = 0 · pile 14 แถวว่าง · web 15 ช่องไม่มีรูป · frozen touching ถูก 'เลือก' ด้วยผลตัวนับเก่า และ P0 audit เปิดภาพ frozen 397 ภาพก่อน split"),
 ("B3", "PASS", "คำนวณซ้ำจาก predictions_old/new.csv ตรง summary ทุกค่า (boos MAE 0.390/0.399 · ontoum =1: 1.000/0.018 · empty 1.000/0.207 · dup 0/346 คู่)"),
 ("C1", "FAIL", "Empty แยก group จริง (tune 145 / report 160) และ score() ใช้ tune-half เท่านั้น · แต่ report-half ถูกรัน 2 ครั้ง (00:45 → 0.8938, แก้โค้ด, 10:03 → 0.9125) · ontoum 895 / boos 598 / count-dev 3 แถว เป็นชุดเดียวกันทั้ง tune และ report · ontoum target = 1 ถูก"),
 ("C2", "FAIL", "146/160 = 0.9125 < 0.95 · ผิด 14 ภาพ ทุกภาพ n=1 จาก 2 ใน 6 วิดีโอ · pattern = วัตถุสว่างอมเหลืองที่มุมบนซ้ายของภาพ (ชิดขอบ, 0.03–0.23% ของภาพ, L*≈72–80, hue≈96°) ไม่ใช่เงาหรือลายพื้น"),
 ("C3", "FAIL", "ทั้งสองอย่าง: fixture ไม่สมจริง (สีเรียบ, ปิดภาพ 92.7%) และตัวนับแตกเมล็ดจริง: 1.7–3.2 ชิ้น/เมล็ด, ref_area 70–228 px เทียบของจริง 1303 px, เมล็ดเข้ม ~105 เมล็ดไม่ถูกนับเลย, n แกว่ง 514↔974 ตามคุณภาพ JPEG · คู่ IoU>0.5 แค่ 4–22 คู่ = ไม่ใช่นับซ้ำ"),
 ("C4", "PASS", "summary เขียน count_test_touching/pile = 'unmeasured', passed = null, eligible = false · ไม่ได้อ้างว่าผ่าน · (ข้อค้นพบเพิ่ม: ภาพกองจริง agtron ผ่านทาง FW → n=1 'exact' ไม่มี warning 16/20 ภาพ)"),
 ("C5", "PASS", "เกิน max_beans → เก็บ 1000 ก้อนใหญ่สุด + method=estimated + note max_beans_cap (ทดสอบ max_beans=100 บน 300 เมล็ด → 100 estimated) · feature/core/bbox เป็น vectorized · ยกเว้น _blob_shapes เป็น loop ต่อทุก component รวมฝุ่น (counter.py:168-183, 430)"),
 ("D1", "PASS", "คำนวณใหม่จาก feature cache + โมเดลราย fold ได้ตรง summary ทั้ง 12 ค่า (cm ต่างสูงสุด 8e-12) · weight 1/n ต่อภาพถูก · หมายเหตุ: ไม่มีไฟล์ prediction รายเมล็ด มีแต่ count รายภาพ"),
 ("D2", "PASS", "0 group ข้าม source · 0 md5/phash ซ้ำข้าม source · 0 แถว frozen · 0 แถว split=test · scaler = สถิติถ่วงน้ำหนักของ train fold เท่านั้น (ต่าง 1e-14) · C และ gap เลือกจาก train fold · ข้อสังเกต: fold rf_boos ไม่มี medium เลย"),
 ("D3", "FAIL", "feature รายเมล็ดไม่ใช่สาเหตุ (ontoum ΔL_med +0.34 ± 1.19) · ตัวโมเดลรายเมล็ดคือสาเหตุ: เอาน้ำหนัก B1 มาใช้กับ feature รายเมล็ดได้ 0.8615 (ontoum) / 0.6306 (boos) · rf_robusta ป้อน 'เมล็ด' ที่ไม่ใช่เมล็ด (L*≈46 ทั้ง medium/dark) เข้าเทรน 74% ของ medium"),
 ("D4", "FAIL", "20.9% มาจาก 2 fold: rf_robusta dark→light 174.4 ภาพ (50.3%) + rf_boos light→dark 158.3 ภาพ (44.0%) · ontoum 3.9% · agtron 1.8%"),
 ("D5", "FAIL", "วัดแค่ 3 เฟรมของวิดีโอเดียว (GT = YOLO prefill): error ต่อภาพ B1 8.33 / B-bean 5.00 / +group 8.33 จาก 6 เมล็ด · ข้อมูลที่ใช้ได้โดยไม่แตะ frozen: rf_boos Mixed 102 เฟรม, rf_robusta mixed 981 ภาพ"),
 ("E1", "PASS", "0 invariant ผิด ใน (100 + 6 + 8 ภาพ) × 3 โมเดล · ส่วนต่าง proportions ≤ 1.5e-4 มาจากการปัด 4 ตำแหน่งตามออกแบบ · เสมอ 9–11/100 ตัดสินด้วยค่าเฉลี่ย prob"),
 ("E2", "PASS", "เกิด 0/100 ทั้งสองโมเดลรายเมล็ด · ถ้าเกิดจะเป็น status 'error' (ไม่ใช่ bad_image) ข้อความ 'ระบบขัดข้องชั่วคราว…' (api.py:167-177) · ในทางปฏิบัติ majority_probs โยนก่อนใน backend → ถอยเป็น B1 เงียบ ๆ"),
 ("E3", "FAIL", "Firmware/service.py:56 บน origin/main บังคับ set(result) == 12 key → ผล v3 (16 key) ถูกปฏิเสธ 100/100 ภาพ + 8/8 ผลเสีย → subscriber ทิ้ง event → SQLite/หน้า market ไม่ได้ข้อมูล · HTTP ตอบ browser ยังปกติ"),
 ("E4", "FAIL", "ที่ 69c4354 สะอาด: ปกติ 320 passed / 15 failed · จำลอง Pi 289 passed / 15 failed / 9 skipped · 15 ตัวที่ล้มอยู่ใน tests/test_firmware_v3.py (ทดสอบ patch FW ที่ยังไม่ commit) · เลขใน commit (335 · 304+9) ได้เฉพาะ worktree ที่มี patch"),
 ("F1", "FAIL", "git apply --check ผ่านทั้ง 6 patch บน 69c4354 ไม่ชน · แต่แตะ 6 ไฟล์ ไม่ใช่ 4 (app.py, service.py, script.js, style.css, index.html, market.html)"),
 ("F2", "PASS", "DOM ใช้ textContent/createElement ทั้งหมด ไม่มี innerHTML · upload สูงสุด 32 MiB (client + MAX_CONTENT_LENGTH) · SQL ใช้ ? ทุกจุด · ไม่มี credential ใน patch · ข้อสังเกต: BEGIN IMMEDIATE ทุกครั้งที่ connect แม้ตอนอ่าน"),
 ("F3", "UNVERIFIED", "อ่านโค้ดแล้วสเกลถูก (sx = canvas/image_size แยกแกน · รูปที่แสดงคือ blob เดียวกับที่ส่ง · DPR ≤ 2 · ResizeObserver) · วาด 964 กรอบ 1.3–3.3 ms บน notebook · ไม่ได้ทดสอบบนมือถือจริง/EXIF จริง"),
 ("F4", "PASS", "mqtt_result() ตัด beans ออกสำหรับ v3 (patch app.py) · enqueue = queue.put_nowait · publish ทำใน daemon thread (mqtt_pub.py:21-43) · ยังไม่เคยส่งถึง broker จริง"),
 ("F5", "FAIL", "252 ms = รอบ pilot LAN บน notebook Windows (loopback proxy, ภาพสังเคราะห์ 4–123 KB) · hotspot n=0 เพราะรอบนั้นไม่ได้รัน hotspot · รอบสุดท้ายที่ครบ 2 profile ได้ p95 3983 / 2995 ms = ไม่ผ่าน · ยังไม่มีตัวเลขบน Pi"),
]
MAIN5 = [
 "Backend: b1_linear ชื่อ b1-Lab_hist-C0.01-R2 จาก ML/models/current (สร้าง 8 ต.ค. 14:23 ก่อน P0–P2 ทั้งหมด) ห่อด้วย contract v3",
 "ตัวนับ: ไม่มี — counter.py ไม่ถูกเรียก · n_beans, counts, count_method, proportions = null · ROAST_BEANS ตั้งหรือไม่ตั้งก็ไม่มีผลกับ backend นี้",
 "beans[]: ว่างเสมอ — ไม่ใช่ทั้ง B1 broadcast, B-bean หรือ B-bean+group",
 "label ระดับภาพ: softmax ของ B1 บน feature Lab_hist จากพิกเซลที่ segment() เลือก · low_conf_threshold = 0.0 → status เป็น ok ทุกภาพที่เปิดได้ แม้ภาพว่าง (ได้ 'คั่วเข้ม' + no_beans_detected)",
 "ฝั่ง FW: /api/predict ตอบ browser ได้ แต่ subscriber ของ origin/main ปฏิเสธ event v3 ทุกตัว → SQLite/หน้า market ว่าง · pytest ที่ commit นี้ล้ม 15 ตัว",
]
BLOCKERS = [
 ("main ไม่สอดคล้องกันเอง: ML v3 + FW v2", "ผล v3 ทุกตัวถูก service.py:56 ปฏิเสธ → ตลาด/SQLite ไม่มีข้อมูล และ pytest ล้ม 15 ตัว", "ให้เพื่อน merge patch แค่ 2 ไฟล์ (service.py + app.py) ก็พอให้ persist ได้ · 4 ไฟล์ที่เหลือเป็นหน้าจอ"),
 ("requirement 'รายเมล็ด + นับ' ยังไม่ถูกเสิร์ฟเลย", "default เป็น B1 ระดับภาพ · ไม่มีโมเดลรายเมล็ดตัวไหนผ่านเกณฑ์ และ models/ ของ main ไม่มีโมเดล v3", "ตัดสินใจว่าจะเสิร์ฟอะไรวันจันทร์ (ดูข้อ 4 ด้านล่าง) · ทางเล็กสุดที่ซื่อตรง = ตัวนับ + label ระดับภาพของ B1 กระจายให้ทุกเมล็ด พร้อม warning"),
 ("ภาพกองจริงนับได้ 1 เมล็ด", "agtron เต็มเฟรม 16/20 ภาพ → n=1, count_method='exact', ไม่มี warning · กองสังเคราะห์แตกเป็นชิ้นและทิ้งเมล็ดเข้ม · ไม่มี GT touching/pile เลย", "นับ GT 7 ภาพ pile ใน dev ด้วยมือ + เพิ่มกฎ 'ก้อนเดียวใหญ่มาก → เส้นทาง pile + estimated' แล้ววัดบน dev ก่อน"),
 ("ตัวจำแนกรายเมล็ดแย่กว่า B1 มาก", "light F1 = 0 บน ontoum และ boos · light↔dark 20.9% (เกณฑ์ 5%)", "การทดลองเดียว: เทรน nested LOSO เดิมโดยตัด rf_robusta ออกจากชุดเทรนรายเมล็ด (ประกาศเกณฑ์ก่อนรัน)"),
 ("ยังไม่มีหลักฐาน < 1 s บน Pi", "ML อย่างเดียวบน notebook กับภาพกองจริง 1600 px = 530–683 ms (เกณฑ์ Pi 500 ms) · E2E รอบสุดท้าย 3–4 s ยังไม่รู้สาเหตุ", "รัน tools/pi_bench.sh บน Pi จริงด้วยภาพจริง 1600 px 20 ภาพ + แยกวัด fetch→JSON ออกจากช่วงรอ animation frame"),
]
DECISIONS = [
 "จะเสิร์ฟอะไรวันจันทร์: (ก) B1 ระดับภาพอย่างเดียว บอกตรง ๆ ว่ายังไม่นับ · (ข) ตัวนับ + label ของ B1 กระจายทุกเมล็ด (นับได้ แต่ภาพปนหลายระดับไม่รองรับ) · (ค) รอผลการทดลองตัด rf_robusta ก่อน — ผมแนะนำ (ข) เป็นค่าตั้งต้น และรัน (ค) คู่ขนาน",
 "patch FW แตะ 6 ไฟล์ ไม่ใช่ 4: รับทั้ง 6 หรือให้เพื่อนรับเฉพาะ service.py + app.py ก่อน",
 "prereg_perbean_v2: อนุมัติย้อนหลังพร้อมบันทึก 3 จุดที่เบี่ยง หรือออก v3 · และจะแก้ fold rf_boos ที่ไม่มี medium (เพราะวิดีโอ Medium-1- ถูก freeze) อย่างไร",
 "อนุญาตให้เทรนใหม่ 1 รอบเพื่อยืนยันสมมติฐาน D3 ไหม (รอบตรวจนี้ห้ามเทรน จึงยังไม่ได้รัน)",
 "metal_chroma_min = 16: ได้ Empty tune-half 100% แต่ตัดเมล็ดคั่วเข้ม (chroma 9–13) ออกเมื่อพื้นหลังไม่ขาวสม่ำเสมอ — ยอมแลกหรือไม่",
 "ใครจะนับ GT: pile 7 ภาพ dev + หา touching สำหรับ dev (ตอนนี้ touching อยู่ใน frozen ทั้งหมด)",
 "worktree seed-count มี Firmware แก้ค้าง 6 ไฟล์ที่ Claude รอบก่อนเขียนไว้ (นอกขอบเขต ML/): เก็บไว้ให้เพื่อนดู หรือให้คุณล้างเอง",
 "โฟลเดอร์รายงานนี้ยังไม่ commit (test ที่ commit นี้ไม่ผ่าน และคำสั่งรอบนี้คืออ่านอย่างเดียว) — จะให้ commit ไหม",
]


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c if isinstance(c, Raw) else esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


class Raw(str):
    pass


def badge(v):
    return Raw(f'<span class="b {v.split()[0].lower()}">{esc(v)}</span>')


cur = E["models"]["current(main default)"]
prev = E["models"]["v3_preview(worktree only)"]
cand = E["models"]["bean_candidate(main)"]


def five_json(model):
    rows = []
    for f in model["five"]:
        r = f["result"]
        rows.append({"image": f["image"], "true": f["true"], "status": r["status"], "label": r["label"], "confidence": r["confidence"],
                     "n_beans": r["n_beans"], "counts": r["counts"], "count_method": r["count_method"], "warnings": r["warnings"],
                     "len_beans": r["len(beans)"], "image_size": r["image_size"], "ml_ms_notebook": r["timing_ms"]["total"],
                     "model": r["model"], "fw_origin_main": f["fw_origin_main"]})
    return json.dumps(rows, ensure_ascii=False, indent=1)


def hundred_row(name, m):
    h = m["hundred"]
    return [name, m["backend"], json.dumps(h["status"]), json.dumps(h["violations"], ensure_ascii=False) or "{}", h["count_ties"], h["n_beans_reported"],
            str(h["n_beans_min_med_max"]), str(h["single_bean_conf_min_max"]), h["dup_pairs_iou>0.5"],
            " / ".join(f"{x:.0f}" for x in h["ms_p50_p95_max"]), h["fw_origin_main_rejects"], h["fw_patched_rejects"], m["majority_valueerror_count"], m["bean_path_fallback_count"]]


sw = D["swap_2x2_macroF1_and_perclass[L,M,D]"]
swap_rows = [[k, f'{v["B1model_x_imageFeat(ref)"][0]:.4f}', f'{v["B1model_x_beanFeat"][0]:.4f}', f'{v["beanModel_x_imageFeat"][0]:.4f}',
              f'{v["beanModel_x_beanFeat(P1)"][0]:.4f}', str(v["beanModel_x_beanFeat(P1)"][1]), v["b1_W_norm"], v["bean_W_norm"]] for k, v in sw.items()]
feat_rows = [[r["source"], r["label"], r["images"], r.get("beans", ""), r.get("bean_L_med", ""), r.get("img_L_med", ""), r.get("b1train_L_med", ""),
              r.get("bean_a", ""), r.get("bean_b", ""), r.get("bean_L_p10", ""), r.get("img_L_p10", "")] for r in D["feature_table"]]
o50 = D["ontoum_bean_vs_image_50"]["all"]
o50_rows = [[k, v["mean_bean"], v["mean_img"], v["mean_diff"], v["sd_diff"], v["max_abs_diff"]] for k, v in o50.items()]
share = D["train_weight_share"]
share_rows = []
for fold in ("ontoum224", "rf_boos"):
    for l in ("light", "medium", "dark"):
        s = share[fold][l]
        share_rows.append([fold, l, s["bean_model_image_weight"], json.dumps(s["bean_share_by_source"]), s["b1_rows(video cap 10)"], json.dumps(s["b1_share_by_source"])])
ld = D["light_dark_by_fold"]
ld_rows = [[k, v["light->dark"], v["dark->light"], v["light_imgs"], v["dark_imgs"], f'{ld["share_of_errors"][k]*100:.1f}%'] for k, v in ld["folds"].items()]
pile_rows = [[k, v["n"], v["ref_area_work_px"], v["gt_visible_area_work_px_median"], v["pred_regions_per_gt_bean"]["mean"], v["gt_beans_with_0_regions"],
              v["pred_regions_with_purity<0.8(span >1 bean)"], v["dup_box_pairs_iou>0.5"]] for k, v in C["pile_runs"].items()]
ag = C["agtron_fw_vs_roi"]
emp = C["empty_report_half"]
cap = C["cap_test_4000_dots"]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b675f;--line:#dedad1;--card:#fff;--pass:#1f7a4d;--fail:#b3261e;--unv:#8a6100;--code:#f1eee7}
@media (prefers-color-scheme:dark){:root{--bg:#161513;--fg:#ece9e2;--mut:#a39e93;--line:#34322d;--card:#1e1d1a;--pass:#5fd39a;--fail:#ff8a80;--unv:#f2c14e;--code:#26241f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 "Sarabun","Noto Sans Thai",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:20px 0 6px}p{margin:6px 0}.mut{color:var(--mut)}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card)}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--code);font-weight:600;white-space:nowrap}td{font-variant-numeric:tabular-nums}
.b{display:inline-block;padding:1px 8px;border-radius:10px;font-weight:700;font-size:12px;border:1.5px solid}.pass{color:var(--pass)}.fail{color:var(--fail)}.unverified{color:var(--unv)}
code,pre{font-family:ui-monospace,Consolas,monospace;background:var(--code);border-radius:4px}code{padding:1px 4px;font-size:12.5px}pre{padding:10px;overflow-x:auto;font-size:12px;line-height:1.45}
ol,ul{margin:6px 0;padding-left:22px}li{margin:4px 0}.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.kpi div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px}
.kpi b{display:block;font-size:22px}.note{border-left:3px solid var(--unv);padding:6px 12px;background:var(--card);margin:10px 0}
"""
n_pass = sum(v == "PASS" for _, v, _ in VERDICTS); n_fail = sum(v == "FAIL" for _, v, _ in VERDICTS); n_unv = sum(v == "UNVERIFIED" for _, v, _ in VERDICTS)
H = []
H.append(f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ตรวจอิสระ P0–P2 · 10 ต.ค. 2569</title><style>{CSS}</style></head><body><main>
<h1>รายงานตรวจอิสระ P0–P2</h1>
<p class="mut">นับเมล็ด + ระดับคั่วรายเมล็ด + contract v3 + dashboard · ตรวจที่ main @ 69c4354 · 10 ต.ค. 2569 · อ่านและรันซ้ำเท่านั้น ไม่ได้แก้โค้ด ไม่ได้เทรน ไม่ได้เปิด split=test หรือแถว frozen</p>
<div class="kpi"><div><b class="pass">{n_pass}</b>PASS</div><div><b class="fail">{n_fail}</b>FAIL</div><div><b class="unverified">{n_unv}</b>UNVERIFIED</div></div>
<p><b>สรุป:</b> main ตอนนี้ยังเสิร์ฟ B1 ระดับภาพตัวเดิมของวันที่ 8 ต.ค. — ไม่มีการนับและไม่มีผลรายเมล็ดออกไปถึงผู้ใช้เลย และผล v3 ทุกตัวถูก Firmware บน origin/main ปฏิเสธตอนบันทึกลง SQLite ตัวเลขใน P1 คำนวณซ้ำได้ตรงและไม่พบ leakage แต่เกณฑ์ที่ประกาศไว้ไม่ผ่านเกือบทั้งหมด และพบสาเหตุของ light F1 = 0 ที่ตรวจสอบได้</p>
<h2>1) ตาราง A–F</h2>""")
H.append(table(["ID", "ผล", "หลักฐาน"], [[i, badge(v), e] for i, v, e in VERDICTS]))
H.append("<h2>2) main เสิร์ฟอะไรจริงตอนนี้</h2><ol>" + "".join(f"<li>{esc(x)}</li>" for x in MAIN5) + "</ol>")
H.append('<p class="mut">หมายเหตุ: Firmware/app.py:33 ใช้ค่า default <code>/opt/topgun/ML/models/current</code> ส่วน run_local.sh:9 ใช้ <code>$PROJECT_DIR/ML/models/current</code> · models/ ไม่อยู่ใน git จึงตรวจได้เฉพาะสำเนาบนเครื่องนี้ · ไฟล์ที่อยู่บน Pi จริง = UNVERIFIED</p>')
H.append("<h2>3) Blocker ก่อนวันจันทร์ 18:00</h2>" + table(["#", "ปัญหา", "หลักฐาน", "ทางแก้ที่เล็กที่สุด"], [[i + 1, a, b, c] for i, (a, b, c) in enumerate(BLOCKERS)]))
H.append("<h2>4) สิ่งที่ต้องให้คุณตัดสินใจ</h2><ol>" + "".join(f"<li>{esc(x)}</li>" for x in DECISIONS) + "</ol>")

H.append("<h2>A. สถานะ repo และสิ่งที่ deploy</h2>")
H.append("<h3>A2 ไฟล์แก้ค้าง 12 ไฟล์</h3><p>ตอนตรวจ main สะอาด จึงยืนยันไม่ได้ว่า 12 ไฟล์นั้นต่างแค่ line ending ข้อเท็จจริงที่อ่านได้: ไม่มี <code>.gitattributes</code> และไฟล์ที่ track 241 ไฟล์เก็บเป็น LF ใน index แต่เป็น CRLF ใน working tree ซึ่งเป็นรูปแบบของ <code>core.autocrlf=true</code></p>"
         '<div class="note">ผลข้างเคียงที่ต้องแจ้ง: คำสั่งแรกที่ผมรันบน main คือ <code>git status --porcelain</code> ซึ่ง git อาจ refresh stat cache ใน <code>.git/index</code> เอง ถ้า 12 ไฟล์นั้น "ค้าง" เพราะ mtime เปลี่ยนแต่เนื้อหาเท่าเดิม การ refresh นี้ทำให้มันหายไปได้ ผมไม่ได้รัน checkout / restore / stash / add ใด ๆ บน main</div>'
         "<p>ถ้ามันกลับมาอีก ให้ตรวจเองตามลำดับนี้ (ไม่ทำลายงาน):</p><pre>git status --porcelain\ngit diff --ignore-cr-at-eol --stat      # ว่าง = ต่างแค่ line ending\ngit diff -w --stat                      # ว่าง = ต่างแค่ whitespace\ngit update-index --refresh              # ล้างเฉพาะ stat ที่ค้าง ไม่แตะเนื้อไฟล์</pre>"
         "<p>ทางป้องกันถาวรคือให้เจ้าของ repo เพิ่ม <code>.gitattributes</code> (<code>* text=auto eol=lf</code>) — ไฟล์นี้อยู่นอก ML/ ผมจึงไม่ได้สร้าง</p>")
H.append("<h3>A3 รันจริง 5 ภาพด้วยค่า default (ROASTML_MODEL = models/current, ไม่ตั้ง ROAST_BEANS)</h3><pre>" + esc(five_json(cur)) + "</pre>")
H.append("<p>เทียบ: ภาพชุดเดียวกันกับโมเดลรายเมล็ด preview (อยู่เฉพาะใน worktree seed-count ไม่ได้อยู่ใน main)</p><pre>" + esc(five_json(prev)) + "</pre>")
H.append("<p>ข้อสังเกตจาก preview: ภาพ boos ที่มี 3 ระดับปนกันได้ medium ทั้ง 6 เมล็ดและไม่มี mixed_roast · ภาพ agtron เต็มเฟรมนับได้ 1 เมล็ด · กองสังเคราะห์ 300 ได้ 514</p>")

H.append("<h2>B. P0 + prereg</h2><h3>B1 จุดที่เบี่ยงจาก prereg</h3><ul>"
         "<li>Empty: prereg กำหนด rf_hendi empty 305 ภาพ · P1 แบ่งครึ่งเป็น tune 145 / report 160 แล้วรายงานเฉพาะ 160 (เข้มกว่า prereg)</li>"
         "<li>rf_boos beans-mode: prereg กำหนดทุกภาพ trainval (P0 audit มี 925) · P1 ใช้ 598 เพราะตัด group ที่ freeze ออก</li>"
         "<li>macro-F1 ราย fold: fold rf_boos เหลือแค่ light 198 + dark 298 ภาพ (medium 0) เพราะวิดีโอ Medium-1- ถูก freeze → macro-F1 3 คลาสมีเพดาน 0.667</li></ul>"
         "<p>เกณฑ์ mixed_dev (candidate ต้องต่ำกว่า B1) ไม่อยู่ใน <code>GATES</code> ของ tools/perbean_protocol.py:14-16 จึงไม่ถูกบันทึกใน declaration</p>")
H.append("<h3>B2 count-test</h3>" + table(["split", "scene", "ที่มา", "กรอกแล้ว", "มีรูป", "จำนวนแถว"], [
    ["dev", "flat", "rf_boos Mixed (YOLO prefill)", "ใช่", "ใช่", 3], ["dev", "pile", "agtron 6 + rf_hendi 1", "ไม่", "ใช่", 7], ["dev", "—", "web slot", "ไม่", "ไม่", 5],
    ["frozen", "flat", "rf_boos", "ใช่", "ใช่", 4], ["frozen", "touching", "rf_boos Medium-1-", "ใช่", "ใช่", 4], ["frozen", "pile", "agtron/rf_hendi", "ไม่", "ใช่", 7], ["frozen", "—", "web slot", "ไม่", "ไม่", 10]]))
H.append("<ul><li>split สุ่มระดับ group ด้วย seed 20261009 · count_test.csv และ count_test_meta.json mtime 9 ต.ค. 22:56 และไม่ถูกแก้อีกเลย → split มาก่อนการนับด้วยคนแน่นอน เพราะยังไม่มีใครนับ</li>"
         "<li>group ที่ freeze: rf_boos 3 วิดีโอ (397 ภาพ), agtron dev2 + dev4 (189 ภาพ), rf_hendi 3 วิดีโอ (68 ภาพ) รวม 654 ภาพ trainval</li>"
         "<li>P1 ไม่แตะ frozen: feature cache 3,892 record อยู่ใน frozen 0 · ค้น path ของ frozen/test ในไฟล์ผล P1 ทุกไฟล์ = 0</li>"
         "<li>แต่ P0 audit (22:44–22:48 ก่อน split 22:56) รันตัวนับทั้งสองบน rf_boos ทั้ง 995 ภาพ รวม 397 ภาพที่ภายหลังถูก freeze และ 8 แถว frozen ของ count-test · แถว touching ถูกเลือกด้วย <code>--touching-from predictions_old.csv</code> (tools/make_count_test.py:79-83) คือเลือกเฟรมที่ตัวนับเก่านับน้อยกว่า GT</li>"
         "<li>B1 ที่ deploy อยู่ (current) เทรนด้วย trainval ทั้งหมดรวม group ที่ freeze → แถว frozen เป็น in-sample สำหรับ label ระดับภาพ</li>"
         "<li>ในรอบตรวจนี้ผมอ่านเฉพาะคอลัมน์ path/split/scene และดูว่าช่อง n_total ว่างหรือไม่ของแถว frozen ไม่ได้เปิดรูปและไม่ได้รันโมเดลบนแถวเหล่านั้น</li></ul>")

H.append("<h2>C. P1 ตัวนับ</h2><h3>C1 ลำดับเวลาของการรัน</h3>" + table(["เวลา (10 ต.ค.)", "run", "Empty zero-rate", "boos MAE", "หมายเหตุ"], [
    ["00:33", "counter_wip_dev", "tune 0.008 (n=120)", "0.058 (n=120)", "ค่าเริ่มต้น"], ["00:38", "counter_grid_dev", "tune (n=120): chroma8 0.375 → chroma12 0.908 → chroma16 1.000", "0.050 (n=120)", "เลือก chroma16"],
    ["00:45", "counter_report_dev", "report-half 0.8938 (n=160)", "0.0769 (n=598)", "เปิด report-half ครั้งที่ 1"], ["09:56", "counter_touch_resume", "tune 1.000 (n=145)", "0.0619 (n=598)", "config เดิมแต่ค่าเปลี่ยน = โค้ดตัวนับถูกแก้"],
    ["09:59", "counter_final_grid", "tune 1.000 (n=145)", "0.0385 (n=598)", "เลือก sep2"], ["10:03", "counter_final_report", "report-half 0.9125 (n=160)", "0.0385 (n=598)", "เปิด report-half ครั้งที่ 2"]]))
H.append("<p>tune-half ได้ 145/145 แต่ report-half ได้ 146/160 — ช่องว่างนี้บอกว่าค่า chroma ที่จูนพอดีกับวิดีโอฝั่ง tune · ค่าที่เลือก (metal_chroma_min=16, ws_separation=2) ต่างจากค่า default ใน CountConfig (8.0, 1.0) ดังนั้น model_card ที่ไม่มี count_config จะได้พฤติกรรมอีกแบบ (เห็นได้จาก bean_candidate ใน main ที่นับภาพ Empty ได้ 5 เมล็ด)</p>")
H.append(f"<h3>C2 ภาพ Empty ที่นับผิด {emp['fail']} ภาพ</h3><p>ดูภาพ overlay ครบทั้ง 14 ภาพ: เป็นภาพภายในถังคั่วโลหะสีเข้ม ทุกภาพที่ผิดมีวัตถุสีอ่อนอมเหลืองโผล่ที่มุมบนซ้ายของเฟรมและถูกนับเป็น 1 เมล็ดแบบ flat/exact ผ่านเส้นทาง local_background · กระจุกอยู่ใน 2 วิดีโอ: {esc(json.dumps(emp['fail_groups']))} จาก 6 วิดีโอ · ภาพที่ผ่านจากวิดีโอเดียวกันมีวัตถุเดียวกันแต่เล็กกว่าเกณฑ์พื้นที่</p>"
         "<p class='mut'>ข้อจำกัดของชุดนี้: 'Empty' ของ rf_hendi คือถังคั่วเปล่า ไม่ใช่โต๊ะหรือกระดาษเปล่าแบบที่จะเจอตอนสาธิต</p>")
H.append("<h3>C3 กองสังเคราะห์ 300 เมล็ด</h3>" + table(["อินพุต", "n ที่นับได้", "ref_area (px)", "พื้นที่เมล็ดจริง (px)", "ชิ้น/เมล็ด", "เมล็ดที่ไม่ถูกนับ", "ชิ้นที่คร่อม >1 เมล็ด", "คู่กรอบ IoU>0.5"], pile_rows))
H.append("<p>ภาพเดียวกันนับได้ 514 หรือ ~970 ขึ้นกับการบีบอัด JPEG อย่างเดียว (นี่คือที่มาของ 964 ใน E2E ซึ่ง browser บีบซ้ำ) · เมล็ดสีเข้มของ fixture (RGB 50,32,22) ไม่ถูกนับเลยเพราะ chroma ต่ำกว่า metal_chroma_min=16 · เมล็ดสีอ่อน/กลางถูกแบ่งเป็นหลายชิ้นซ้อนกัน · label เป็น partition ไม่ทับกัน จึงไม่ใช่การนับซ้ำด้วยกรอบซ้อน</p>"
         "<p>หลักฐานเดียวกันในข้อมูลจริง: rf_robusta ภาพ dark ตรวจไม่เจอเมล็ดเลย 149/500 (30%) เทียบ medium 32/479 (7%)</p>")
H.append(f"<h3>C4 ภาพกองจริงผ่านเส้นทางของ FW</h3><p>agtron 20 ภาพ (สุ่ม seed 20261010, ไม่ใช่ frozen) ส่งทั้งเฟรมผ่าน predict_bytes: n_beans = {esc(str(ag['fw_n_beans']))} · ภาพเดียวกันผ่าน ROI + force_pile (เส้นทางที่ P1 ใช้ประเมิน) ได้ {min(ag['roi_n_beans'])}–{max(ag['roi_n_beans'])} เมล็ด</p>"
         "<p>สาเหตุจากโค้ด: กองบนจานบนพื้นขาวกินพื้นที่ภาพไม่ถึง 60% จึงไม่เข้าเงื่อนไข pile (counter.py:403, 415) → ทั้งกองเป็นก้อนเดียวและเป็นก้อนเดียวที่ใช้กำหนดขนาดอ้างอิง → นับเป็น 1 เมล็ดแบบ exact (counter.py:437-441) ผลของ fold agtron ใน P1 จึงไม่ได้สะท้อนสิ่งที่ FW จะได้</p>")
H.append(f"<h3>C5 cap และ loop</h3><p>finish() ที่ counter.py:385-393 ตัดเหลือก้อนใหญ่สุด max_beans ก้อน เปลี่ยน method เป็น estimated → ผู้ใช้ได้ warning count_visible_only + bean_count_estimated แต่ไม่มี warning เฉพาะว่า 'ถึงเพดาน' · ภาพจุดเล็ก 4,000 จุดที่ 1600×1200: ขั้น count ใช้ {cap['api_stage_ms']['count']} ms และรวม {cap['api_total_ms']} ms บน notebook เพราะ _blob_shapes วน Python ต่อทุก component</p>")

H.append("<h2>D. P1 ตัวจำแนกรายเมล็ด</h2><h3>D1 ตาราง paired (คำนวณใหม่ = summary ทุกค่า)</h3>" + table(["fold", "วิธี", "macro-F1 (ถ่วง 1/n)", "ต่างจาก B1", "ผ่าน −0.02", "F1 light / medium / dark", "light↔dark"], [
    ["ontoum224", "B1 broadcast", "0.8579", "—", "อ้างอิง", "0.905 / 0.772 / 0.897", "0.0034"], ["ontoum224", "B-bean", "0.4034", "−0.4544", "ไม่ผ่าน", "0.000 / 0.420 / 0.790", "0.0235"], ["ontoum224", "B-bean+group", "0.4034", "−0.4544", "ไม่ผ่าน", "0.000 / 0.420 / 0.790", "0.0235"],
    ["rf_robusta", "B1 broadcast", "0.3683", "—", "อ้างอิง", "0.092 / 0.524 / 0.489", "0.1202"], ["rf_robusta", "B-bean", "0.2396", "−0.1287", "ไม่ผ่าน", "0.016 / 0.380 / 0.323", "0.4346"], ["rf_robusta", "B-bean+group", "0.2634", "−0.1050", "ไม่ผ่าน", "0.014 / 0.447 / 0.329", "0.3977"],
    ["rf_boos", "B1 broadcast", "0.6234", "—", "อ้างอิง", "0.872 / 0.000 / 0.998", "0.0000"], ["rf_boos", "B-bean", "0.2634", "−0.3600", "ไม่ผ่าน", "0.000 / 0.000 / 0.790", "0.3192"], ["rf_boos", "B-bean+group", "0.2505", "−0.3728", "ไม่ผ่าน", "0.000 / 0.000 / 0.752", "0.3972"],
    ["agtron", "B1 broadcast", "0.6374", "—", "อ้างอิง", "0.611 / 0.714 / 0.587", "0.0000"], ["agtron", "B-bean", "0.6534", "+0.0159", "ผ่าน", "0.460 / 0.644 / 0.856", "0.0300"], ["agtron", "B-bean+group", "0.7919", "+0.1545", "ผ่าน", "0.667 / 0.787 / 0.922", "0.0000"]]))
H.append("<p class='mut'>ภาพที่ตรวจไม่เจอเมล็ดถูกตัดออกจากทั้งสองฝั่ง (rf_robusta 183/1046) · fold agtron ประเมินด้วย ROI + force_pile ซึ่ง FW ไม่มี · ตาราง per-class ใน report เดิมใส่ F1 รายคลาสแบบถ่วงน้ำหนักคู่กับ macro-F1 แบบไม่ถ่วงในคอลัมน์สุดท้าย (เช่น rf_robusta 0.3613)</p>")
H.append("<h3>D3 วินิจฉัย light F1 = 0</h3><p><b>ขั้นที่ 1 — feature รายเมล็ดกับ feature ทั้งภาพของ B1 บนภาพ ontoum เดียวกัน 50 ภาพ (สุ่ม seed 20261010, ทุกภาพ n=1)</b></p>" + table(["feature", "รายเมล็ด", "ทั้งภาพ (B1)", "ส่วนต่างเฉลี่ย", "SD", "ต่างสูงสุด"], o50_rows))
H.append("<p>แทบเท่ากัน → WB และ core mask ไม่ใช่สาเหตุบนภาพเรียบ · WB ใช้ PreparedImage ตัวเดียวกันทั้ง segment และตัวนับ (train_perbean.py:47-49, bean_backend.py:53-69) · เทรนและ serve เรียก count_features ตัวเดียวกัน (perbean.py:16-19) · parity test มีจริง (tests/test_beans.py:61, 124) แต่ครอบเฉพาะฉากสังเคราะห์แบบ flat ไม่ครอบเส้นทาง pile/ROI</p>")
H.append("<p><b>ขั้นที่ 2 — สลับโมเดลกับ feature (ใช้โมเดลราย fold ที่มีอยู่ ไม่ได้เทรน)</b></p>" + table(["fold", "น้ำหนัก B1 × feature ภาพ", "น้ำหนัก B1 × feature เมล็ด", "โมเดลเมล็ด × feature ภาพ", "โมเดลเมล็ด × feature เมล็ด (P1)", "F1 L/M/D ของ P1", "‖W‖ B1", "‖W‖ เมล็ด"], swap_rows))
H.append("<p>บน ontoum และ rf_boos น้ำหนักของ B1 ทำงานกับ feature รายเมล็ดได้ดีเท่าเดิม ส่วนโมเดลรายเมล็ดแย่ไม่ว่าจะป้อน feature แบบไหน → ปัญหาอยู่ที่สิ่งที่โมเดลรายเมล็ดเรียนมา</p>")
H.append("<p><b>ขั้นที่ 3 — โมเดลรายเมล็ดเรียนจากอะไร</b> (ค่าเฉลี่ยต่อ source × คลาส)</p>" + table(["source", "คลาส", "ภาพ", "เมล็ด", "L_med เมล็ด", "L_med ภาพ", "L_med ที่ B1 ใช้เทรน", "a เมล็ด", "b เมล็ด", "L_p10 เมล็ด", "L_p10 ภาพ"], feat_rows))
H.append("<p>rf_robusta: 'เมล็ด' ที่ตัวนับหาได้มี L_med ≈ 46 ทั้ง medium และ dark ขณะที่ B1 เทรนจากกล่อง YOLO ได้ 26.3 และ 14.4 → บริเวณที่ตัวนับจับไม่ใช่เมล็ดที่ label พูดถึง แต่ถูกป้อนเข้าเทรนพร้อม label ของภาพ</p>")
H.append("<p><b>ขั้นที่ 4 — สัดส่วนน้ำหนักในชุดเทรนของ fold ที่ light F1 = 0</b></p>" + table(["fold ที่กันออก", "คลาส", "น้ำหนักรวม (ภาพ)", "สัดส่วนต่อ source — โมเดลเมล็ด", "แถวเทรนของ B1 (cap 10 เฟรม/วิดีโอ)", "สัดส่วนต่อ source — B1"], share_rows))
H.append("<p><b>สมมติฐาน (เรียงตามน้ำหนักหลักฐาน)</b></p><ol>"
         "<li>H1: rf_robusta ปนเปื้อนชุดเทรนรายเมล็ด — ใน fold ontoum มันคือ 74% ของน้ำหนัก medium และ 43% ของ dark โดยมี L_med ≈ 46 → โมเดลเรียนว่า 'สว่างราว 46 = medium/dark' → ontoum light (L_med 46.8) ถูกทายเป็น medium ทั้ง 300 ภาพ และ boos light (L_med 47.3, chroma ต่ำ) ถูกทายเป็น dark 158/198</li>"
         "<li>H2: ไม่มี cap เฟรมต่อวิดีโอในโมเดลรายเมล็ด — rf_boos (3 วิดีโอ) เป็น 63% ของน้ำหนัก light ใน fold ontoum และ C ที่เลือกได้ (1.0) ให้ ‖W‖ ใหญ่กว่า B1 2–3 เท่า</li>"
         "<li>H3: เส้นทาง pile ตัดร่องมืดออกก่อนคิดสี — agtron L_med รายเมล็ดสูงกว่าระดับภาพ ~6 และ L_p10 สูงกว่า ~12 (agtron ไม่ได้ข่มด้วยจำนวน เพราะถ่วง 1/n แล้วเหลือ 16–26% ต่อคลาส)</li></ol>"
         "<p><b>การทดลองเล็กที่สุดที่ยืนยันได้ (ยังไม่ได้รัน):</b> รัน tools/train_perbean.py เดิมทุกอย่าง แต่ตัด rf_robusta ออกจาก <code>arrays(train)</code> ของโมเดลรายเมล็ดเท่านั้น (B1 broadcast คงเดิม) ประกาศก่อนรันว่า H1 ถูกถ้า light F1 ของ fold ontoum ≥ 0.70 และ light→dark ของ fold rf_boos ≤ 5% · ถ้าไม่ถึง ให้ทดสอบ H2 ด้วย cap 10 เฟรม/วิดีโอ</p>"
         "<p class='mut'>ตัวเลข 'น้ำหนัก B1 × feature เมล็ด' ในตารางขั้นที่ 2 เป็นการวินิจฉัยบน dev ไม่ใช่ผลของ candidate ที่ประกาศไว้ล่วงหน้า ห้ามใช้เป็นหลักฐานผ่านเกณฑ์</p>")
H.append("<h3>D4 light↔dark ของ B-bean แยกตาม fold</h3>" + table(["fold", "light→dark (ภาพ)", "dark→light (ภาพ)", "ภาพ light", "ภาพ dark", "ส่วนแบ่งของ error"], ld_rows) + f"<p>รวม {ld['pooled_rate']:.4f} ตรงกับ report เดิม (0.2087)</p>")

H.append("<h2>E. P2 contract v3</h2><h3>E1–E3 รัน 100 ภาพสุ่มจาก trainval (ตัด group ที่ freeze, seed 20261010) + 6 ภาพว่าง/สังเคราะห์ + 8 อินพุตเสีย</h3>")
H.append(table(["โมเดล", "backend", "status", "invariant ผิด", "เสมอ", "มี n_beans", "n min/med/max", "conf ภาพ 1 เมล็ด", "คู่ IoU>0.5", "ML ms p50/p95/max (notebook)", "FW origin ปฏิเสธ", "FW+patch ปฏิเสธ", "ValueError majority", "ถอยเป็น B1"],
               [hundred_row("current (default)", cur), hundred_row("bean_candidate (main)", cand), hundred_row("v3 preview (worktree)", prev)]))
H.append("<p>invariant ที่ตรวจ: key ครบและเรียงตาม RESULT_KEYS · json.dumps ได้ · n_beans == len(beans) == sum(counts) · counts ตรงกับ label ของ beans · label อยู่ในกลุ่มที่ count สูงสุด · label == argmax(probs) · proportions รวม 1 และ null เมื่อ n=0 · bbox อยู่ใน image_size · mixed_roast / no_beans_detected / bean_count_estimated ตรงกับ counts และ count_method · ผลเสียต้องเป็น null ทั้งชุด</p>")
H.append("<p>อินพุตเสีย: ไบต์ว่าง, ไบต์สุ่ม, JPEG ตัดท้าย, ข้อความ, PDF, None → bad_image · PNG 4×4 → bad_image พร้อมข้อความ 'รูปเล็กเกินไป' · ส่ง str → error</p>")
H.append("<p><b>ข้อค้นพบที่ไม่ใช่ invariant แต่กระทบผู้ใช้</b></p><ul>"
         "<li>confidence ของโมเดลรายเมล็ด = 0.999 × สัดส่วนคลาสข้างมาก + 0.001 × prob เฉลี่ย (perbean.py:86) → ภาพ 1 เมล็ดได้ 0.9995–1.0 ทั้ง 29 ภาพไม่ว่าตัวจำแนกจะลังเลแค่ไหน และ low_conf_threshold ของทุก card = 0.0 → ไม่มีทางได้ status low_confidence</li>"
         "<li>ภาพว่าง (ขาวล้วน, ดำล้วน, noise, ถังเปล่า): status ok พร้อม label 'คั่วเข้ม/คั่วกลาง' ความมั่นใจ 0.87–0.94 และ warning no_beans_detected — ข้อความขึ้นว่า 'ผลวิเคราะห์: คั่วเข้ม'</li>"
         "<li>เมื่อเส้นทางรายเมล็ดพัง backend จับ exception แล้วคืนผล B1 พร้อม n_beans = null (bean_backend.py:106-108) ผู้ใช้ไม่รู้ว่าระบบถอย · ใน 100 ภาพเกิด 0 ครั้ง</li></ul>")
H.append("<h3>E4 pytest</h3>" + table(["ที่ไหน", "ชุด", "ผล"], [
    ["69c4354 สะอาด (worktree ตรวจ)", "ปกติ", "320 passed, 15 failed"], ["69c4354 สะอาด (worktree ตรวจ)", "จำลอง Pi (-p tools.pi_runtime_plugin)", "289 passed, 15 failed, 9 skipped"],
    ["worktree seed-count (มี patch FW)", "ปกติ", "335 passed"], ["worktree seed-count (มี patch FW)", "จำลอง Pi", "304 passed, 9 skipped"]]))
H.append("<p>15 ตัวที่ล้มอยู่ใน tests/test_firmware_v3.py ทั้งหมด (เช่น <code>AttributeError: module 'app' has no attribute 'mqtt_result'</code>) · 9 ตัวที่ skip: sklearn 7 (test_compare_tabular, test_d1_tools, test_export_d1, test_roi_scaling, test_run_tools, test_baseline ×2) และ onnx/onnxruntime 2 (test_onnx_backend) · เลขของ commit P1 (308 · 277+9) ไม่ได้รันซ้ำ = UNVERIFIED</p>")

H.append("<h2>F. Firmware patch (ยังไม่ commit)</h2><ul>"
         "<li>F1: patch อยู่ที่ results/fw_v3_review_20261010/diff/Firmware/ · รวม 332 บรรทัดเพิ่ม / 167 บรรทัดลบ (service.py เปลี่ยน 125 บรรทัด, script.js เปลี่ยน 315 บรรทัด) · ตรวจด้วย <code>git apply --check</code> เท่านั้น ไม่ได้ apply จริง</li>"
         "<li>F2: connect_db() ของ patch รัน CREATE TABLE + BEGIN IMMEDIATE ทุกครั้ง และ read_market_snapshot() อ่านทุกแถวไม่มี LIMIT ขณะที่หน้าเว็บเรียก /api/market ทุก 3 วินาทีต่อ client → บน Pi จะแย่ง CPU/lock กับ /api/predict เมื่อข้อมูลสะสม</li>"
         "<li>F3: ถ้า browser เปิดไฟล์ไม่ได้ (เช่น HEIC บน Chrome) จะส่งไฟล์ต้นฉบับและซ่อน canvas → ไม่มี overlay แต่ผลตัวเลขยังแสดง</li>"
         "<li>F4: payload MQTT ของ v3 ไม่มี key beans เลย · subscriber ที่ patch แล้วรับได้ทั้งแบบมีและไม่มี beans</li></ul>")
H.append("<h3>F5 E2E ทั้ง 3 รอบที่มีผล</h3>" + table(["รอบ (เวลา)", "LAN n", "LAN p50 / p95 ms", "hotspot n", "hotspot p50 / p95 ms", "p95 < 1 s"], [
    ["fw_v3_e2e_20261010_final (11:39)", 20, "229.0 / 252.0", 0, "—", "LAN ผ่าน · hotspot ไม่ได้รัน"],
    ["fw_v3_e2e_20261010_review (11:43)", 20, "2990.9 / 2992.5", 0, "—", "ไม่ผ่าน"],
    ["fw_v3_e2e_20261010_visible (11:49)", 20, "2988.4 / 3982.6", 20, "2988.6 / 2995.2", "ไม่ผ่านทั้งคู่"]]))
H.append("<p>report.html เดิมรายงานรอบ 11:49 เป็นผลสุดท้ายและเขียนว่าไม่ผ่าน ส่วน 252 ms ถูกระบุเป็น pilot · ค่า ~2990 ms คงที่ทุกฉากทั้งที่ ML ใช้ 114–280 ms แปลว่าเวลาส่วนใหญ่ไม่ได้อยู่ที่ ML หรือเครือข่ายจำลอง — เข้ากับการที่ requestAnimationFrame ถูกหน่วงเมื่อ iframe/แท็บไม่ได้วาดจริง แต่ผมไม่ได้พิสูจน์ข้อนี้</p>")
H.append("<p>ทั้งสามรอบวัดบน notebook Windows 11 ผ่าน proxy บน loopback ที่หน่วง RTT และจำกัด bitrate เฉพาะ /api/predict ใช้ภาพสังเคราะห์ 900×600 ขนาด 4–123 KB และปิด broker MQTT → ใช้สรุปเรื่อง Pi ไม่ได้ · วัดเพิ่มในรอบตรวจนี้ (ML อย่างเดียวบน notebook, ภาพ agtron จริง): ไฟล์ต้นฉบับ 7 MB ใช้ 805–1010 ms · ย่อเหลือ 1600 px ก่อนส่งใช้ 530–683 ms (B1 ตัวปัจจุบัน 249–318 ms)</p>")

H.append("<h2>สิ่งที่ไม่ได้ตรวจ / UNVERIFIED</h2><ul>"
         "<li>origin/main บน GitHub จริง (ไม่ได้ fetch) และโมเดลที่อยู่บน Pi จริง</li><li>core.autocrlf และสาเหตุของ 12 ไฟล์แก้ค้าง</li>"
         "<li>เลข pytest ของ commit P0 และ P1 · ผลบน Pi จริงทุกชนิด (latency, RAM, HEIC)</li><li>overlay บนมือถือจริง/ภาพ EXIF จริง · การส่ง MQTT ถึง broker จริง</li>"
         "<li>สาเหตุของ E2E ~2990 ms · สมมติฐาน H1–H3 ของ D3 (ต้องเทรนใหม่จึงยืนยันได้)</li><li>notes ของแถว frozen ใน count_test.csv (ไม่ได้อ่าน)</li></ul>")
H.append("<h2>ทำซ้ำได้อย่างไร</h2><p>สคริปต์และหลักฐานดิบอยู่ในโฟลเดอร์เดียวกับรายงานนี้ รันจาก <code>ML/</code> ด้วย <code>ROAST_DATA_DIR</code> ชี้ไปที่โฟลเดอร์ข้อมูล:</p>"
         "<pre>python results/review_p0_p2_20261010/scripts/run_e.py   results/review_p0_p2_20261010/evidence_contract.json   # A3, E1–E3\n"
         "python results/review_p0_p2_20261010/scripts/d_verify.py                                                        # D1, D2\n"
         "python results/review_p0_p2_20261010/scripts/d3.py      results/review_p0_p2_20261010/evidence_perbean.json    # D3, D4\n"
         "python results/review_p0_p2_20261010/scripts/c23.py     results/review_p0_p2_20261010/evidence_counter.json    # C2–C5\n"
         "python -m pytest -q -p no:cacheprovider [-p tools.pi_runtime_plugin]                                            # E4</pre>"
         "<p class='mut'>seed 20261010 · Python 3.14.3, numpy 2.5.3, OpenCV 5.0.0, Pillow 12.3.0 (pytest ใช้ ML/.venv ของ main: sklearn 1.9.1, pytest 9.1.1) · ภาพ overlay ที่สร้างจากข้อมูลอยู่นอก repo ที่ <code>$ROAST_DATA_DIR/contact_sheets/review_p0_p2_20261010/</code> (C2_empty_fail_overlay.jpg, C3_synthetic_pile_overlay.jpg, C3_pile_crop_zoom.png, C4_agtron_fullframe_overlay.jpg)</p>")
H.append("</main></body></html>")
(out / "report.html").write_text("\n".join(H), encoding="utf-8")

summary = {
    "review": "P0-P2 independent review", "date": "2026-10-10", "scope_commit": "69c4354", "seed": 20261010,
    "rules_followed": {"code_modified": False, "trained_or_tuned": False, "split_test_opened": False, "count_test_frozen_rows_run": False,
                       "commits": 0, "main_side_effect": "git status on main may have refreshed .git/index stat cache; nothing else"},
    "counts": {"PASS": n_pass, "FAIL": n_fail, "UNVERIFIED": n_unv},
    "verdicts": [{"id": i, "verdict": v, "evidence": e} for i, v, e in VERDICTS],
    "main_serves_now": MAIN5,
    "blockers": [{"rank": i + 1, "problem": a, "evidence": b, "smallest_fix": c} for i, (a, b, c) in enumerate(BLOCKERS)],
    "decisions_needed": DECISIONS,
    "key_numbers": {
        "pytest_at_69c4354": {"normal": {"passed": 320, "failed": 15}, "pi_sim": {"passed": 289, "failed": 15, "skipped": 9}},
        "pytest_with_fw_patch": {"normal": {"passed": 335}, "pi_sim": {"passed": 304, "skipped": 9}},
        "fw_origin_main_rejects_v3_of_100": cur["hundred"]["fw_origin_main_rejects"],
        "invariant_violations": {k: m["hundred"]["violations"] for k, m in E["models"].items()},
        "majority_valueerror_of_100": {k: m["majority_valueerror_count"] for k, m in E["models"].items()},
        "empty_report_half": {"n": emp["n"], "fail": emp["fail"], "zero_rate": emp["zero_rate"], "fail_groups": emp["fail_groups"]},
        "synthetic_pile_n_by_input": {k: v["n"] for k, v in C["pile_runs"].items()},
        "agtron_fullframe_fw_n_beans": ag["fw_n_beans"], "agtron_roi_n_beans": ag["roi_n_beans"],
        "swap_macro_f1": {k: {kk: vv[0] for kk, vv in v.items() if isinstance(vv, list)} for k, v in sw.items()},
        "light_dark_by_fold": ld,
        "e2e_runs_ms": {"final_1139": {"LAN_p95": 252.0, "hotspot_n": 0}, "review_1143": {"LAN_p95": 2992.5, "hotspot_n": 0}, "visible_1149": {"LAN_p95": 3982.6, "hotspot_p95": 2995.2}},
        "notebook_ml_ms_real_agtron": {"orig_7MB_bean_model": [805, 1010], "pre1600_bean_model": [530, 683], "pre1600_b1_current": [249, 318]},
    },
    "evidence_files": ["evidence_contract.json", "evidence_perbean.json", "evidence_counter.json", "pytest_69c4354_normal.txt", "pytest_69c4354_pi.txt", "scripts/"],
    "overlays_outside_repo": "$ROAST_DATA_DIR/contact_sheets/review_p0_p2_20261010/",
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", out / "report.html", (out / "report.html").stat().st_size, "bytes")
