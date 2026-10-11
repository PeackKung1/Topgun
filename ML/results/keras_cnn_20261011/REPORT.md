# ทดลอง CNN ด้วย Keras สำหรับจำแนกระดับคั่วระดับภาพ — 11 ต.ค. 2026

**ไม่มี arm ไหนชนะ B1** ตามกฎตัดสินเดิม B1 (mean LOSO macro-F1 0.6382) ยังเป็นตัว deploy
จึงไม่ได้ทำขั้น export ONNX และวัด latency

protocol ทำซ้ำ `tools/train_d1.py` (groupguard) เปลี่ยนแค่ตัวโมเดลเป็น Keras
แต่มีข้อมูลตั้งต้น 2 อย่างที่ไม่ใช่ไฟล์ของ 8 ต.ค. (ดูหัวข้อ "สิ่งที่ต่างจากเกณฑ์เดิม") จึงยังอ้างไม่ได้ว่าตรงเกณฑ์เดิมทุกจุด

## ผลหลัก

mean LOSO macro-F1 = เฉลี่ย 4 fold แล้วเฉลี่ยข้าม 3 seed ± std (ddof=1)

| โมเดล | mean LOSO macro-F1 | pooled macro-F1 | fold แย่สุด (seed แรก) | cross-step | ECE | ชนะ B1 |
|---|---|---|---|---|---|---|
| **B1** (อ้างอิง) | **0.6382** ± 0.0000 | 0.6646 | rf_robusta 0.385 | 0.031 | 0.232 | — |
| K2_new_finetune | 0.4693 ± 0.0259 | 0.5064 | rf_robusta 0.363 | 0.060 | 0.219 | ไม่ |
| K1_control_finetune | 0.4660 ± 0.0336 | 0.5081 | rf_boos 0.462 | 0.066 | 0.259 | ไม่ |
| D1_finetune (อ้างอิง, PyTorch) | 0.4389 ± 0.0194 | 0.5213 | rf_boos 0.283 | 0.083 | 0.237 | ไม่ |
| D1_finetune_medaug (อ้างอิง) | 0.4114 ± 0.0415 | 0.5367 | rf_boos 0.303 | 0.085 | 0.249 | ไม่ |
| K1_control_frozen | 0.3573 ± 0.1140 | 0.4186 | rf_boos 0.304 | 0.110 | 0.235 | ไม่ |

แถวอ้างอิงมาจาก `results/summary_20261008/model_comparison.csv` ไม่ได้รันใหม่
ตัวเลขของ K1/K2 คำนวณจากไฟล์ predictions ด้วย `tools.summarize_runs.evaluate` (โค้ดเดียวกับตาราง 8 ต.ค.)

### ราย fold (macro-F1 เฉลี่ย 3 seed ± std)

| fold (n) | B1 | D1_finetune | K1_control_finetune | K2_new_finetune | K1_control_frozen |
|---|---|---|---|---|---|
| ontoum224 (895) | 0.890 | 0.502 ± 0.129 | 0.525 ± 0.202 | 0.525 ± 0.202 | 0.449 ± 0.208 |
| rf_robusta (1046) | 0.385 | 0.521 ± 0.036 | 0.485 ± 0.048 | 0.456 ± 0.094 | 0.382 ± 0.146 |
| rf_boos (893) | 0.666 | 0.262 ± 0.077 | 0.361 ± 0.167 | 0.401 ± 0.203 | 0.219 ± 0.074 |
| agtron (561) | 0.612 | 0.471 ± 0.024 | 0.494 ± 0.026 | 0.495 ± 0.033 | 0.380 ± 0.094 |
| pooled (3395) | 0.665 | 0.521 ± 0.027 | 0.508 ± 0.044 | 0.506 ± 0.042 | 0.419 ± 0.102 |

ตารางเต็ม (acc, cross-step, ECE ราย fold รวม D1_finetune_medaug) อยู่ใน `compare_per_fold.csv`

## อ่านผลอย่างไร

- **K1 เทียบ D1 (Keras ≈ PyTorch หรือไม่):** K1_control_finetune 0.466 ± 0.034 เทียบ D1_finetune 0.439 ± 0.019
  ต่างกัน 0.027 ซึ่งเล็กกว่า std ข้าม seed ของ K1 เอง ด้วย 3 seed จึงแยกไม่ออกว่าต่างกันจริง
  ผลสอดคล้องกับที่คาดว่า Keras ให้ผลระดับเดียวกับ PyTorch แต่ไม่ใช่การพิสูจน์ว่าเท่ากัน
  (น้ำหนัก ImageNet และ preprocessing ของสอง framework เป็นคนละชุด)
- **K2 เทียบ K1 (augmentation `wbexp` ช่วยหรือไม่):** ต่างกัน 0.003 เล็กกว่า std ข้าม seed เกือบสิบเท่า run นี้ไม่แสดงว่า `wbexp` ช่วย
  และการทดสอบนี้อ่อน เพราะ inner validation เลือก `wbexp` แค่ fold เดียว (rf_boos) อีก 3 fold เลือก `none`
  ใน fold rf_boos เอง K2 ได้ 0.401 ± 0.203 เทียบ K1 0.361 ± 0.167 ซึ่งอยู่ในช่วงความแกว่งของ seed
- **ความแกว่งข้าม seed สูงมาก:** config เดียวกันบน fold ontoum224 ได้ macro-F1 0.539 / 0.719 / 0.316 ใน 3 seed
  ผลของ fold เดี่ยวจาก seed เดียวจึงไม่ควรใช้สรุปอะไร
- **inner validation สูงกว่า holdout มาก:** F1 บน inner validation ของ fine-tune อยู่ที่ 0.84–0.96 ใน 3 fold
  แต่ holdout ของ source ที่ไม่เคยเห็นได้ 0.36–0.53 เป็นรูปแบบเดียวกับที่บันทึกไว้ของ D1 เมื่อ 8 ต.ค.
- **K1_control_frozen:** head ถูกเทรนแค่ 4–13 ก้าว (1 epoch = 1 full-batch step ตาม protocol ของ D1) ผลจึงขึ้นกับค่าเริ่มต้นของ head
  ตรงกับหมายเหตุ "head unconverged" ของ D1_frozen_v1 (0.319 ± 0.036) ในตาราง 8 ต.ค.

## config ที่ inner validation เลือก (จูนด้วย seed 20261006 ครั้งเดียว)

| arm | fold | config | epoch | temperature | inner F1 ของตัวที่เลือก |
|---|---|---|---|---|---|
| K1_control_frozen | ontoum224 | lr 0.003 | 13 | 1.0 | 0.426 |
| | rf_robusta | lr 0.01 | 6 | 0.75 | 0.779 |
| | rf_boos | lr 0.01 | 6 | 0.75 | 0.787 |
| | agtron | lr 0.01 | 4 | 0.75 | 0.715 |
| K1_control_finetune | ontoum224 | none | 4 | 1.5 | 0.602 |
| | rf_robusta | mild | 3 | 0.5 | 0.864 |
| | rf_boos | mild | 1 | 1.0 | 0.879 |
| | agtron | mild | 3 | 1.0 | 0.959 |
| K2_new_finetune | ontoum224 | none | 4 | 1.5 | 0.602 |
| | rf_robusta | none | 4 | 2.0 | 0.847 |
| | rf_boos | wbexp | 1 | 0.75 | 0.908 |
| | agtron | none | 2 | 0.5 | 0.955 |

trial ทั้งหมดของ inner อยู่ใน `<arm>_<fold>_selection.json`

## สิ่งที่ต่างจากเกณฑ์เดิม (บันทึกใน `plan.json` ก่อนเทรนครั้งแรก)

1. **inner group ไม่ใช่ไฟล์ของ 8 ต.ค.** เครื่องนี้ไม่มี `cache/qa_audit_20261008/inner_groups.json`
   จึงสร้างใหม่ด้วย `tools.qa_audit` (policy และ threshold เดิม) ไว้ที่ `cache/qa_audit_regen_20261011/`
   ได้ 1,344 group เดิม → 1,326 inner group ยืนยันไม่ได้ว่าตรงกับไฟล์เดิม inner split จึงอาจต่างจากของ D1
2. **manifest.csv เป็นตัวที่สร้างใหม่จาก zip 4 source** (ไม่มี rf_hendi) จำนวนต่อ source/label ตรงกับ `manifest_summary.json`
   แต่ sha256 ไม่ตรงของเดิม จึงยืนยันไม่ได้ว่า pixel ของ view ตรงกับของ D1 ทุกไบต์
3. **รันบน CPU** TensorFlow ไม่มี build ที่ใช้ GPU บน Windows โดยตรงตั้งแต่ 2.11 และเครื่องนี้ไม่มี WSL2
4. **K1 ใช้น้ำหนัก ImageNet และ preprocessing ของ Keras** ซึ่งต่างจาก torchvision
5. **มีการรันทดสอบโค้ดกับข้อมูลชุดย่อย** (`--smoke-limit 150`) หลังเขียน plan และก่อนรันจริง ผลอยู่นอก `results/` และไม่ได้ใช้เลือกอะไร

เกณฑ์ที่ตรวจแล้วว่าตรง: 3,395 ภาพ trainval (895 / 1,046 / 893 / 561 ต่อ fold), view จาก `prepare_views`,
`video_cap_mask(seed=20261006)` เหลือ 2,393 ภาพ (train ต่อ fold 1,498 / 1,516 / 2,333 / 1,832), ไม่ได้อ่าน split=test

## K2 คืออะไร

เปลี่ยนตัวแปรเดียวจาก K1_control_finetune: ใน grid แทน `mild` ด้วย `wbexp`
(`mild` ที่ขยายเฉพาะ `exposure_EV` เป็น ±0.5 และ `linear_red_blue_gain_log` เป็น ±0.2 ซึ่งเป็นค่าของ `medium` ที่ประกาศไว้เดิม
ส่วน rotation, flip, crop 0.85–1.0, noise, blur, JPEG คงเป็น mild) backbone, lr, จำนวน epoch และขนาด grid เท่า K1
ไม่มี hue rotation, saturation, grayscale หรือ equalization

เหตุผลที่เขียนไว้ก่อนรัน: หลักฐาน 8 ต.ค. ระบุว่า D1 พลาดที่ rf_boos เพราะ colour cast แรง
และ arm medium เดิมเปลี่ยนความแรงของแสง/สีพร้อมกับ crop 0.5 และ JPEG q30 จึงแยกผลไม่ออก

## Environment

| | |
|---|---|
| Python | 3.12.1 (venv แยกที่ `ML/venv/tf`) |
| TensorFlow / Keras | 2.21.0 / 3.15.1 (backend tensorflow) |
| CUDA / cuDNN | ไม่มี (build ของ Windows เป็น CPU, `built_with_cuda = False`) |
| GPU | RTX 4060 มีในเครื่องแต่ไม่ได้ใช้ |
| CPU | AMD Ryzen 5 5600, RAM 16 GB, Windows 11 |
| numpy / opencv / scikit-learn | 2.5.3 / 5.0.0.93 / 1.9.1 (เท่ากับ env ของ 8 ต.ค.) |
| น้ำหนัก | `weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5` sha256 `002803a7…39c5f6` |

TensorFlow รองรับ Python 3.10–3.13 (ตรวจจากเอกสารติดตั้งและ wheel บน PyPI) เครื่องนี้ใช้ 3.12

## การรัน

- เวลารวมราว 101 นาที: รอบแรก 70 นาที (หยุดตามที่ผู้ใช้สั่งที่ 26/36 หน่วย) แล้วรันต่ออีก 31 นาที
- fine-tune ใช้ราว 10 นาทีต่อ seed (K1) และ 7 นาทีต่อ seed (K2) สำหรับ 4 fold
- **การหยุดแล้วรันต่อไม่เปลี่ยนผล:** fold ontoum224 ของ K1 และ K2 เลือก config/epoch/temperature เดียวกัน
  ความน่าจะเป็นของทั้ง 895 ภาพตรงกันทุกค่า (max |Δp| = 0) ทั้ง 3 seed โดย K1 เทรนก่อนหยุด และ K2 seed 20261007/20261008 เทรนหลังรันต่อ
- ไฟล์ predictions ทั้ง 9 ไฟล์: 3,395 แถว เรียงตาม path คอลัมน์ตามที่กำหนด ความน่าจะเป็นรวมกันได้ 1 (คลาดเคลื่อนสูงสุด 2e-16)

คำสั่งที่รันจริง (จาก `D:\Topgun\ML`, `ROAST_DATA_DIR=D:\data`):

```
.venv\Scripts\python -m tools.train_d1 --prepare-views --workers 4                                   (76 s)
.venv\Scripts\python -m tools.qa_audit --result-dir results/qa_audit_regen_20261011 --no-contact-sheets --workers 4   (110 s)
.venv\Scripts\python -m tools.qa_audit --result-dir results/qa_audit_regen_20261011 --build-inner-groups

py -3.12 -m venv venv\tf
venv\tf\Scripts\python -m pip install tensorflow==2.21.0
venv\tf\Scripts\python -m pip install -e ".[train]"

venv\tf\Scripts\python -m tools.train_keras_cnn --run-dir results/keras_cnn_20261011 ^
    --inner-groups D:\data\cache\qa_audit_regen_20261011\inner_groups.json --write-plan --deviation "..." (5 ข้อ)
venv\tf\Scripts\python -m tools.train_keras_cnn --run-dir <scratch>\keras_cnn_smoke --inner-groups <เดิม> --smoke-limit 150
venv\tf\Scripts\python -m tools.train_keras_cnn --run-dir results/keras_cnn_20261011 ^
    --inner-groups D:\data\cache\qa_audit_regen_20261011\inner_groups.json                           (รัน 2 ครั้ง: เริ่ม + รันต่อ)
```

## ไฟล์

| ไฟล์ | เนื้อหา | อยู่ใน git |
|---|---|---|
| `plan.json` | แผนที่เขียนก่อนเทรน รวม arm, grid, เหตุผลของ K2, deviation | ใช่ |
| `report.json`, `compare.csv`, `compare_per_fold.csv` | ผลสรุปและตารางเทียบ | ใช่ |
| `<arm>_<fold>_selection.json` | trial ทั้งหมดของ inner (12 ไฟล์) | ใช่ |
| `run_state.json` | ผลราย seed/fold พร้อม trace ที่ใช้สรุป | ใช่ |
| `<arm>_<seed>_predictions.csv` | ผลรายภาพ (9 ไฟล์) | ไม่ (gitignore) |
| `ML/models/candidates/keras_cnn_20261011/` | น้ำหนักของทุก fold/seed | ไม่ (gitignore) |
| `results/qa_audit_regen_20261011/` | policy และสรุปของ QA ที่สร้างใหม่ (ไม่รวม mask รายแถว) | ใช่ |

## ข้อจำกัด

- LOSO ระดับ source บนชุดข้อมูลที่หาได้ ไม่ใช่หลักฐานบน Pi, ระบบจริง หรือชุด test ที่แช่แข็ง
- 3 seed น้อยเกินกว่าจะแยกผลต่างระดับ 0.03 ระหว่าง arm ของ CNN ออกจากความแกว่งได้
- ถ้าต้องการอ้างว่าเทียบกับ D1 ด้วยเกณฑ์เดิมทุกจุด ต้องรันซ้ำด้วย `inner_groups.json` และ `manifest.csv` ตัวจริงของ 8 ต.ค.
