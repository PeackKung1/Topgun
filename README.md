# Topgun

ระบบจำแนกระดับคั่วเมล็ดกาแฟจากรูปสำหรับบอร์ด Raspberry Pi โดยให้ผู้ใช้เชื่อมต่อ Wi-Fi hotspot ของบอร์ด แล้วเปิดหน้าเว็บจากมือถือ

## Firmware ที่มีใน repo

- `Firmware/app.py` ให้หน้าเว็บ `/`, API `POST /api/predict`, ประวัติ `/market` และ health check `/health`
- หน้าเว็บย่อรูปเป็น JPEG ด้านยาวไม่เกิน 1600 px ก่อนอัปโหลด และจับเวลาตั้งแต่เริ่มส่งจนแสดงผลด้วย `performance.now()`
- `Firmware/mqtt_pub.py` ใส่ผลทำนายลง bounded queue; worker แยกส่ง MQTT ผ่าน HiveMQ Cloud/TLS จึงไม่รอ broker ใน request path
- `Firmware/service.py` รับ event จาก MQTT ตรวจ contract แล้วบันทึก SQLite โดย `msg_id` เป็น primary key เพื่อกันข้อมูลซ้ำ
- `Firmware/deploy/` มี unit files, broker config, hotspot script และ health check หลัง boot

## สถานะและข้อจำกัด

- `ML/SPEC.md` และ model backend ยังไม่มีใน checkout นี้ การเชื่อมต่ออิง contract ปัจจุบันจาก `ML/roastml/api.py` และ `contract.py` เท่านั้น
- ค่าเริ่มต้น `ROASTML_MODEL=stub` ใช้เปิดเว็บและลอง flow ได้; ตั้ง path โมเดลจริงหลัง ML มี backend ที่โหลดได้
- `Firmware/deploy/coffee_service.service` ใช้ `/opt/topgun`, service account `topgun` และ port `8080`; ยืนยันและปรับให้ตรง Pi ก่อนติดตั้ง
- ตั้ง hotspot ด้วย NetworkManager ต้องระบุ wireless interface และ SSID/password ใน environment บน Pi ก่อนเรียกสคริปต์
- ค่า HiveMQ host/port/topic/TLS ตั้งต้นอยู่ใน `.env.example`; กำหนด username/password ใน `/etc/topgun/topgun.env` บน Pi และอย่า commit รหัสผ่าน
- MQTT event queue อยู่ใน RAM และมีขนาดจำกัด จึงไม่เก็บ event ค้างไว้เมื่อเครื่องดับหรือ broker ไม่กลับมา
- เวลา end-to-end ที่หน้าเว็บรายงานครอบคลุมการย่อภาพ, upload, inference, response และ render; ต้องวัดบน Pi จริงเพื่อยืนยันเป้าหมาย 1 วินาที

## ติดตั้ง Python แบบ offline

วาง wheels ที่ตรงกับ OS, Python และ CPU architecture ของ Pi ไว้ใน `wheelhouse/` แล้วรัน `./setup.sh` ใน repo การเตรียม wheelhouse และติดตั้ง systemd/NetworkManager ต้องทำหลังตรวจรุ่น Pi และ OS จริง

## โครงสร้างโปรเจกต์

```text
Topgun/
├── Firmware/                     # เพื่อน
│   ├── app.py                    # /, POST /api/predict, /market, /health
│   ├── mqtt_pub.py               # queue → MQTT (ไม่อยู่ใน request path)
│   ├── service.py                # subscribe → validate → SQLite (UNIQUE msg_id)
│   ├── templates/                # index.html, market.html
│   ├── static/                   # script.js (ย่อรูป + performance.now) — ไม่ใช้ CDN
│   ├── deploy/                   # coffee_service.service, mosquitto.conf, hotspot.sh
│   ├── requirements-fw.txt
│   └── tests/
├── ML/                           # เรา
│   ├── roastml/                  # api.py, stub.py, preprocess.py, model.py
│   ├── models/current/           # model.onnx + model_card.json
│   ├── scripts/                  # make_manifest, train, export, bench_pi
│   ├── tests/
│   ├── requirements-pi.txt
│   └── requirements-train.txt
├── setup.sh                      # offline install ทั้งสองฝั่ง
├── .env.example
├── .gitignore
└── README.md
```
