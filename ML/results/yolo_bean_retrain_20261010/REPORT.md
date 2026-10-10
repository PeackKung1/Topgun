# เทรน detector นับเมล็ด (YOLO11n) ใหม่ — 10 ต.ค. 2026

**NOT A REPRODUCTION** ของ `results/yolo_bean_20261010/final_v2`

config การเทรนตรงกับ run เดิมทุกค่า แต่ข้อมูลที่ใช้ไม่ครบ: เครื่องนี้ได้รับแค่ zip ต้นทาง 4 ไฟล์
(`agtron`, `ontoum224`, `rf_boos`, `rf_robusta`) ไม่มี `rf_hendi`, `manifest.csv`, `count_test/`,
`selected_config.json` และ `det_b1_variants_20261010.zip` เจ้าของเครื่องสั่งให้เทรนจาก 4 ไฟล์นี้

## สรุป

| | run เดิม `final_v2` | run นี้ `final` |
|---|---|---|
| precision (val สังเคราะห์, epoch สุดท้าย) | 0.941 | 0.940 |
| recall | 0.942 | 0.937 |
| mAP50 | 0.985 | 0.984 |
| mAP50-95 | 0.837 | 0.839 |
| เวลาเทรน + export | 1276.7 s (GTX 1660 SUPER) | 816.2 s (RTX 4060) |
| `bean.onnx` | 10,624,034 bytes | 10,624,000 bytes |
| train / val | 2036 / 100 | 2036 / 100 |
| ภาพลบจริง (rf_hendi Empty) | มี (ประมาณ 145) | **0** |
| ภาพสังเคราะห์จาก sprite ontoum | ประมาณ 1100 | **1245** |

ตัวเลข val มาจากภาพสังเคราะห์ (และ val ของสอง run เป็นภาพคนละชุดกัน) ใช้ดูได้แค่ว่าเทรนจบตามปกติ
สรุปไม่ได้ว่าโมเดลใหม่ดีขึ้นหรือแย่ลง

## สิ่งที่ต่างจาก run เดิม

1. **ไม่มีภาพลบจาก rf_hendi** `negatives = 0` และไม่มีพื้นหลังจริงสำหรับภาพสังเคราะห์ (`real_bgs` ว่าง)
   ยอด train ถูกคงไว้ที่ 2036 ด้วยการเพิ่มภาพสังเคราะห์ ontoum เป็น 1245 ตามสูตร `n_ontoum = 1245 - negatives`
   โมเดลนี้จึงไม่เคยเห็นภาพเครื่องคั่วเปล่าของจริง ต้องวัด `empty_zero_rate` ใหม่ก่อนนำไปใช้
2. **`manifest.csv` สร้างขึ้นใหม่** จาก 4 source ด้วย `tools.make_manifest.build()` (ตัด rf_hendi)
   sha256 ไม่ตรงกับของเดิมอยู่แล้วเพราะไม่มีแถว rf_hendi แต่จำนวนแถวต่อ source/split/label, จำนวน group
   และเครื่อง test ของ agtron ตรงกับ `results/manifest_summary.json` ทุกตัว (5315 แถว) ดู `manifest_4src_check.json`
3. **`count_test/` เป็น stand-in** มีแค่รายชื่อ frozen group ที่คัดลอกจาก `frozen_groups_excluded`
   ของ declaration เดิม (8 group) ไม่มีภาพ count dev/frozen ใช้เทรนได้อย่างเดียว ประเมิน gate ไม่ได้
   frozen group ของ rf_boos และ agtron ทั้ง 5 group มีอยู่ใน manifest ใหม่และถูกกันออกจาก `DevData.rows` เหมือนเดิม
4. **config ตัวนับสร้างขึ้นใหม่** `selected_config_reconstructed.json` =
   `CountConfig(metal_chroma_min=16.0, ws_separation=2.0).to_dict()` มาจากบันทึกการ review ไม่ใช่ไฟล์ต้นฉบับ
   จำนวน sprite ที่ได้ตรงกับของเดิม (ontoum224 1186 / rf_boos 255)
5. **environment** Python 3.13.5 (เดิม 3.14.3), RTX 4060 (เดิม GTX 1660 SUPER), torch 2.14.1+cu130
   (run เดิมไม่ได้บันทึกเวอร์ชัน torch) numpy / opencv / pillow / ultralytics ตรงกับของเดิม
6. **ไม่ได้เทียบกับโมเดลเดิม** ไม่มี `det_b1_variants_20261010.zip` จึงยังไม่ได้เทียบ `n_beans` / label ผ่าน roastml

## Environment

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26300 |
| GPU | NVIDIA GeForce RTX 4060 8 GB, driver 610.88 |
| Python | 3.13.5 (venv `ML/.venv`) |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 (CUDA 13.0, cuDNN 9.24) |
| ultralytics | 8.4.175 (`YOLO_AUTOINSTALL=false`) |
| numpy / opencv / pillow | 2.5.3 / opencv-python-headless 5.0.0.93 / 12.3.0 |
| onnx / onnxruntime | 1.23.2 / 1.30.0 |
| git | branch `retrain/yolo-bean-20261010` จาก `main` @ `0546b41` |

## ผลตรวจก่อนเทรน

- **hash โค้ด** 5 ไฟล์ที่กำหนดตรงกับ `source_sha256` ของ declaration เดิม:
  `roastml/counter.py`, `roastml/segment.py`, `tools/perbean_protocol.py` ตรงแบบ CRLF ตามที่อยู่บนดิสก์,
  `tools/make_synth_beans.py`, `tools/train_yolo_bean.py` ตรงเมื่อแปลงเป็น LF
  (อีก 7 ไฟล์ใน declaration ไม่ตรงเพราะถูกแก้หลัง revision `69c4354`: `roastml/api.py`, `det_backend.py`,
  `detector.py`, `perbean.py`, `tools/build_det_model.py`, `eval_det_beans.py`, `select_bean_labels.py`)
- **ขั้น A** ไฟล์ภาพของ `DevData.rows` (4643 แถว) มีครบ 100% ทุก source: agtron 372, ontoum224 1194,
  rf_boos 598, rf_robusta 2479 · rf_hendi 0 แถว ดู `data_check.json`
- **ขั้น C** `negatives = 0`, `real_boos = 441` (120 + 120 + 102 + 99) →
  `n_ontoum = 1245`, `n_boos = 350`, `n_val = 100`
- **ขั้น D** `summary.json` ของ `yolo_synth_retrain`: sprites ontoum224 1186 / rf_boos 255,
  final train 2036, holdout_boos train 1245, val 100 ·
  sets: synth_ontoum 1245, synth_boos 350, real_boos 441, negatives 0, synth_val 100

## ผลเทรน `final`

- 30 epochs, batch 8, imgsz 640, seed 20261009, single_cls, amp=False, deterministic, ใช้ `last.pt`
- 816.2 s รวม export ONNX · VRAM สูงสุดที่ Ultralytics รายงาน 2.95 GB
- val epoch สุดท้าย (monitor only): P 0.9398, R 0.9375, mAP50 0.9839, mAP50-95 0.8386
- `pytest tests/test_detector.py`: 46 passed
- smoke test: โหลด `bean.onnx` ผ่าน `roastml.detector.BeanDetector` แล้วรันบนภาพ val สังเคราะห์ 8 ภาพได้ผลปกติ
- Ultralytics เตือนว่าภาพมีได้ถึง 501 object แต่ `max_det=300` ตอน val (ค่า default เหมือน run เดิม)
  ตัวเลข val ของภาพกองแน่นจึงอาจถูกจำกัด
- โมเดลอยู่ที่ `ML/models/yolo_bean_retrain_20261010/final/` (`bean.pt`, `bean.onnx`) ไม่เข้า git
  `bean.onnx` sha256 `b25e7a67ac2ce31a2b3adcc33196ab57127b04218cbc4c032a9a92f0bcc0d8a0`

## ผลเทรน `holdout_boos`

เทรนจากภาพสังเคราะห์ของ sprite ontoum อย่างเดียว (1245 ภาพ, ไม่มี negatives) ไม่เคยเห็น rf_boos
config เดียวกับ `final` · run เดิมที่เทียบคือ `results/yolo_bean_20261010/holdout_boos_v2`

| | `holdout_boos_v2` เดิม | run นี้ |
|---|---|---|
| precision (val สังเคราะห์, epoch สุดท้าย) | 0.944 | 0.935 |
| recall | 0.938 | 0.943 |
| mAP50 | 0.985 | 0.983 |
| mAP50-95 | 0.835 | 0.836 |
| เวลาเทรน + export | 773.7 s | 527.9 s |

- ตัวเลขนี้เป็น val สังเคราะห์เช่นกัน ยังไม่ได้วัดบน rf_boos จริง (`tools.eval_det_beans`)
- โมเดลอยู่ที่ `ML/models/yolo_bean_retrain_20261010/holdout_boos/` ไม่เข้า git
  `bean.onnx` 10,624,007 bytes sha256 `4e01b924e4792cb64e3c7bba180af271f0d1c102d3e7ceb24805128363c3688c`

## คำสั่งที่รันจริง

ทุกคำสั่งรันจาก `D:\Topgun\ML` ด้วย `ROAST_DATA_DIR=D:\data` และ `YOLO_AUTOINSTALL=false`

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[train,dev]"
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install ultralytics==8.4.175
.venv\Scripts\python -m pip install onnx onnxruntime==1.30.0
.venv\Scripts\python -m pip uninstall -y opencv-python opencv-python-headless
.venv\Scripts\python -m pip install opencv-python-headless

git switch -c retrain/yolo-bean-20261010
(hardlink D:\data\<source>.zip -> D:\data\zips\<source>.zip ทั้ง 4 ไฟล์)
.venv\Scripts\python -m tools.extract_zips --data-dir D:\data --only agtron ontoum224 rf_boos rf_robusta
.venv\Scripts\python results\yolo_bean_retrain_20261010\scripts\build_manifest_4src.py          (124 s)
.venv\Scripts\python results\yolo_bean_retrain_20261010\scripts\make_count_test_standin.py
.venv\Scripts\python results\yolo_bean_retrain_20261010\scripts\check_data_and_args.py

.venv\Scripts\python -m tools.make_synth_beans --name yolo_synth_retrain ^
    --count-config results\yolo_bean_retrain_20261010\selected_config_reconstructed.json ^
    --n-ontoum 1245 --n-boos 350 --n-val 100                                                    (347 s)
(ดาวน์โหลด yolo11n.pt ด้วย ultralytics.utils.downloads.attempt_download_asset -> models\pretrained\)
.venv\Scripts\python -m tools.train_yolo_bean --exp final --dataset yolo_synth_retrain ^
    --out results/yolo_bean_retrain_20261010/final --epochs 30 --batch 8 --imgsz 640 --device 0 (816 s)
.venv\Scripts\python -m pytest tests/test_detector.py -q
.venv\Scripts\python -m tools.train_yolo_bean --exp holdout_boos --dataset yolo_synth_retrain ^
    --out results/yolo_bean_retrain_20261010/holdout_boos --epochs 30 --batch 8 --imgsz 640 --device 0 (528 s)
```

`yolo11n.pt` มาจาก `github.com/ultralytics/assets/releases/download/v8.4.0/` 5,613,764 bytes
sha256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | เนื้อหา |
|---|---|
| `final/declaration.json`, `final/train_summary.json` | ประกาศก่อนเทรนและผลเทรนของ `final` |
| `holdout_boos/declaration.json`, `holdout_boos/train_summary.json` | ประกาศก่อนเทรนและผลเทรนของ `holdout_boos` |
| `selected_config_reconstructed.json` | config ตัวนับที่สร้างขึ้นใหม่ |
| `manifest_4src_check.json` | ผลเทียบ manifest ใหม่กับ `manifest_summary.json` |
| `data_check.json` | ผลขั้น A และ C |
| `scripts/` | สคริปต์ 3 ตัวที่ใช้สร้าง manifest, count_test stand-in และตรวจข้อมูล |

## ที่ยังต้องทำเมื่อได้ไฟล์ครบ

- เทรนซ้ำด้วย `manifest.csv`, `count_test/` และ rf_hendi ตัวจริง จึงจะเรียกว่า reproduction ได้
- เทียบกับ `det_b1_variants_20261010` ผ่าน roastml (`n_beans`, label)
- วัด gate บนข้อมูลจริง (`tools.eval_det_beans`) โดยเฉพาะ `empty_zero_rate`
- วัด `holdout_boos` บน rf_boos จริง (`boos_mae`)
