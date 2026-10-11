# รวมผล B1 กับ CNN (Keras) แบบน้ำหนักเท่ากัน — 11 ต.ค. 2026

**ไม่มี ensemble ไหนชนะ B1 และทุกตัวแย่กว่า B1 ตัวเดียว** B1 (0.6382) ยังเป็นตัว deploy

กฎรวมผลไม่มีพารามิเตอร์ให้จูน: `p = 0.5 * p_B1 + 0.5 * p_CNN` ต่อภาพ แล้วเลือกคลาสที่ความน่าจะเป็นสูงสุด
`plan.json` เขียนก่อนคำนวณ ensemble ตัวแรก และรายงานครบทั้ง 3 ตัวที่ประกาศไว้

## ผล

mean LOSO macro-F1 = เฉลี่ย 4 fold แล้วเฉลี่ย 3 seed ของ CNN ± std (ddof=1)

| โมเดล | mean LOSO macro-F1 | pooled macro-F1 | cross-step | ECE | ชนะ B1 |
|---|---|---|---|---|---|
| **B1** (อ้างอิง และที่ refit ในรอบนี้) | **0.6382** | 0.6646 | 0.031 | 0.232 | — |
| E_B1+K1_control_finetune | 0.5051 ± 0.0395 | 0.5609 | 0.056 | 0.097 | ไม่ |
| E_B1+K2_new_finetune | 0.4976 ± 0.0381 | 0.5621 | 0.053 | 0.092 | ไม่ |
| E_B1+K1_control_frozen | 0.4568 ± 0.1030 | 0.5072 | 0.087 | 0.109 | ไม่ |
| K2_new_finetune (สมาชิก) | 0.4693 ± 0.0259 | 0.5064 | 0.060 | 0.219 | ไม่ |
| K1_control_finetune (สมาชิก) | 0.4660 ± 0.0336 | 0.5081 | 0.066 | 0.259 | ไม่ |
| K1_control_frozen (สมาชิก) | 0.3573 ± 0.1140 | 0.4186 | 0.110 | 0.235 | ไม่ |

### ราย fold (macro-F1)

| fold | B1 | E_B1+K1_finetune | E_B1+K2_finetune | E_B1+K1_frozen | K1_finetune เดี่ยว |
|---|---|---|---|---|---|
| ontoum224 | 0.890 | 0.644 ± 0.165 | 0.644 ± 0.165 | 0.570 ± 0.217 | 0.525 ± 0.202 |
| rf_robusta | 0.385 | 0.499 ± 0.053 | 0.437 ± 0.039 | 0.377 ± 0.065 | 0.485 ± 0.048 |
| rf_boos | 0.666 | 0.387 ± 0.196 | 0.415 ± 0.216 | 0.350 ± 0.160 | 0.361 ± 0.167 |
| agtron | 0.612 | 0.490 ± 0.035 | 0.494 ± 0.031 | 0.531 ± 0.012 | 0.494 ± 0.026 |
| pooled | 0.665 | 0.561 ± 0.051 | 0.562 ± 0.051 | 0.507 ± 0.103 | 0.508 ± 0.044 |

## อ่านผลอย่างไร

- **โดยเฉลี่ย ensemble ดีกว่า CNN เดี่ยว แต่แย่กว่า B1** ensemble กับ CNN แบบ fine-tune ชนะ B1 แค่ fold เดียวคือ rf_robusta
  (0.385 → 0.44–0.50) ซึ่งเป็น fold ที่ CNN นำ B1 อยู่ก่อนแล้ว และเสียมากใน ontoum224 (0.890 → 0.644) กับ rf_boos (0.666 → 0.39–0.42)
  ส่วน ensemble กับ CNN แบบ frozen แพ้ B1 ทุก fold
- **CNN กลบ B1 เมื่อเฉลี่ยความน่าจะเป็น** temperature ที่ B1 เลือกบน inner validation คือ 5.0 / 0.5 / 1.5 / 1.0
  ใน fold ontoum224 (T = 5.0) ความน่าจะเป็นของ B1 แบนมาก ผลรวมจึงตามคำตอบของ CNN เกือบทั้งหมด
  ทั้งที่ fold นั้น B1 ถูก 89% และ CNN ถูกราว 57%
- **สองโมเดลพลาดคนละภาพอยู่จริง** ใน pooled มี 16.5–17% ของภาพที่ CNN (fine-tune) ถูกแต่ B1 ผิด
  และ 27–28% ที่ B1 ถูกแต่ CNN ผิด ถ้ารู้ล่วงหน้าว่าควรเชื่อใคร ความแม่นจะได้ถึง 0.83 (B1 เดี่ยว 0.66)
  ตัวเลข 0.83 เป็นเพดานที่ใช้คำตอบจริงช่วยเลือก ทำไม่ได้ในการใช้งาน แต่บอกว่ากฎรวมแบบเฉลี่ยตรง ๆ ใช้ข้อมูลเสริมนี้ไม่ได้
- **ECE ของ ensemble ต่ำลง** (0.09–0.11 เทียบ 0.23) แต่ความถูกต้องแย่ลง จึงไม่ใช่เหตุผลให้เลือกใช้

## ผลพลอยได้: B1 ทำซ้ำได้ตรงทุกค่าบนเครื่องนี้

ก่อนรวมผล B1 ถูก refit ด้วย `tools.compare_tabular.run_loso` และต้องผ่านด่านทำซ้ำค่าอ้างอิง

| ตรวจ | ผล |
|---|---|
| mean LOSO macro-F1 แบบ fixed protocol | 0.6381973706626417 ตรงกับ `baseline_loso.json` ทุกหลัก |
| macro-F1 ราย fold เทียบ `per_fold.csv` ของ 8 ต.ค. | ต่างกันสูงสุด 0.0 |
| ECE ราย fold (ขึ้นกับ temperature ที่เลือกบน inner split) | ต่างกันสูงสุด 0.0 |
| ชื่อ feature cache | `baseline_features_6ae3ccfbdf.json` ตรงกับของอ้างอิง, error 0 ภาพ |

ผลนี้เป็นหลักฐานว่า manifest ที่สร้างใหม่และ inner group ที่สร้างใหม่ให้ผลของ B1 เหมือนของ 8 ต.ค. ทุกค่าที่ตรวจได้
ซึ่งลดข้อกังวลเรื่อง deviation 2 ข้อแรกในรายงาน `keras_cnn_20261011` ลง
แต่ยังไม่ใช่การพิสูจน์ว่าไฟล์ `inner_groups.json` เหมือนกันทุกบรรทัด หรือ pixel ของ view ของ CNN ตรงกันทุกไบต์

## ข้อจำกัด

- **แนวคิดนี้เกิดหลังเห็นผล holdout ของทั้งสองโมเดล** (ว่า CNN นำบน rf_robusta) กฎรวมไม่มีส่วนให้จูน
  แต่การตัดสินใจลองใช้ข้อมูล holdout จึงบันทึกไว้ใน plan ว่าถ้าชนะก็เป็นแค่สมมติฐานที่ต้องยืนยันกับข้อมูลใหม่
- ทดสอบกฎเดียว (น้ำหนักเท่ากัน) น้ำหนักที่จูนบน inner validation หรือการให้ B1 เป็นหลักแล้วใช้ CNN เฉพาะเมื่อ B1 ไม่มั่นใจ
  ยังไม่ได้ลอง และต้องเก็บผลทำนายของ CNN บน inner validation เพิ่ม (รอบนี้ไม่ได้บันทึกไว้)
- LOSO ระดับ source บนชุดข้อมูลเดิม ไม่ใช่หลักฐานบน Pi หรือชุด test ที่แช่แข็ง

## การรัน

```
.venv\Scripts\python -m tools.ensemble_b1_cnn --out results/ensemble_b1_keras_20261011 --cnn-run results/keras_cnn_20261011 ^
    --inner-groups D:\data\cache\qa_audit_regen_20261011\inner_groups.json --write-plan
.venv\Scripts\python -m tools.ensemble_b1_cnn --out results/ensemble_b1_keras_20261011 --cnn-run results/keras_cnn_20261011 ^
    --inner-groups D:\data\cache\qa_audit_regen_20261011\inner_groups.json --workers 4          (101 s รวมสกัด feature 87 s)
```

env: Python 3.13.5, numpy 2.5.3, scikit-learn 1.9.1 (venv `ML/.venv`) ไม่ใช้ GPU

ไฟล์ใน git: `plan.json`, `report.json`, `compare.csv`, `compare_per_fold.csv`, รายงานนี้ และ `tools/ensemble_b1_cnn.py`
ไฟล์ผลรายภาพ (`*_predictions.csv`, `predictions_*.csv`, `training_mask_*.csv`, `metrics_*.json`) ถูก gitignore
