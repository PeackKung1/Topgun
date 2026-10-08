# Topgun

ระบบจำแนกระดับคั่วเมล็ดกาแฟจากรูปสำหรับบอร์ด Raspberry Pi โดยให้ผู้ใช้เชื่อมต่อ Wi-Fi hotspot ของบอร์ด แล้วเปิดหน้าเว็บจากมือถือ

## Firmware ที่มีใน repo

- `Firmware/app.py` โหลด `roastml.api.load(ROASTML_MODEL)` ตอนเริ่ม service แล้วส่ง bytes ที่อัปโหลดให้ `predict_bytes(raw)`; ให้หน้าเว็บ `/`, API `POST /api/predict`, ประวัติ `/market` และ health check `/health`
- หน้าเว็บพยายามย่อรูปเป็น JPEG ด้านยาวไม่เกิน 1600 px; ถ้า browser ย่อไม่ได้จะส่งไฟล์ต้นฉบับแทนเมื่อขนาดไม่เกิน 32 MiB และจับเวลาตั้งแต่เริ่มจนแสดงผลด้วย `performance.now()`
- `Firmware/mqtt_pub.py` ใส่ผลทำนายลง bounded queue; worker แยกส่ง MQTT ผ่าน HiveMQ Cloud/TLS จึงไม่รอ broker ใน request path
- `Firmware/service.py` รับ event จาก MQTT ตรวจ contract แล้วบันทึก SQLite โดย `msg_id` เป็น primary key เพื่อกันข้อมูลซ้ำ
- `Firmware/deploy/` มี unit files, hotspot script และ health check หลัง boot; MQTT ใช้ HiveMQ Cloud ผ่าน TLS โดยตรง

## สถานะและข้อจำกัด

- `ML/SPEC.md` และไฟล์โมเดล `ML/models/current/` ไม่มีใน checkout นี้; Firmware อิง contract ใน `ML/roastml/api.py` และ `contract.py` และค่าเริ่มต้นจะโหลดโมเดลจาก `ML/models/current/`
- `Firmware/run_local.sh` ใช้ `ML/models/current/` เป็นค่าเริ่มต้น; ตั้ง `ROASTML_MODEL=stub` เองได้เฉพาะทดสอบหน้าเว็บ โดยผลเป็นข้อมูลจำลอง ไม่ใช่การวิเคราะห์จริง
- หากโมเดลโหลดไม่ได้ เว็บยังเปิดได้ แต่ `/api/predict` และ `/health` จะตอบ 503; ต้องนำโมเดลที่ ML ส่งมาไว้ใน path ที่ `ROASTML_MODEL` ระบุก่อนวิเคราะห์รูปจริง
- ผลนับรายเมล็ดขึ้นกับ backend ส่ง `beans` กลับมา; backend `linear_backend.py` ใน checkout นี้ยังส่งเฉพาะผลรวมทั้งภาพ จึงยังไม่มีผลแยกทีละเมล็ด
- `Firmware/deploy/coffee_service.service` ใช้ `/opt/topgun`, service account `topgun` และอ่าน port จาก `TOPGUN_PORT` (ค่าเริ่มต้น `8080`)
- ตั้ง hotspot ด้วย NetworkManager ต้องระบุ wireless interface และ SSID/password ใน environment บน Pi ก่อนเรียกสคริปต์
- ค่า HiveMQ host/port/topic/TLS ตั้งต้นอยู่ใน `.env.example`; กำหนด username/password ใน `/etc/topgun/topgun.env` บน Pi และอย่า commit รหัสผ่าน
- MQTT event queue อยู่ใน RAM และมีขนาดจำกัด จึงไม่เก็บ event ค้างไว้เมื่อเครื่องดับหรือ broker ไม่กลับมา
- เวลา end-to-end ที่หน้าเว็บรายงานครอบคลุมการย่อภาพ, upload, inference, response และ render; ต้องวัดบน Pi จริงเพื่อยืนยันเป้าหมาย 1 วินาที

## ติดตั้ง Python แบบ offline

วาง wheels ที่ตรงกับ OS, Python และ CPU architecture ของ Pi ไว้ใน `wheelhouse/` แล้วรัน `./setup.sh` ใน repo ก่อนเตรียม wheelhouse ให้ตรวจ `python3 --version` และสถาปัตยกรรมบน Pi ให้ตรงกับ wheel ที่เตรียมไว้

ข้อควรระวัง: `ML/requirements-pi.txt` เลือก `numpy==2.4.6` สำหรับ Python 3.11 และ `numpy==2.5.3` สำหรับ Python 3.12 ขึ้นไป หากทำ wheelhouse สำหรับ Python 3.11 ต้องมี wheel ของ NumPy รุ่น 2.4.6 ที่ตรงกับสถาปัตยกรรม Pi (เช่น `cp311` และ `aarch64`) ด้วย อย่านำ wheelhouse ของ Python 3.11 ไปใช้กับ Python 3.13 หรือคนละสถาปัตยกรรม

## Checklist ก่อนเปิดใช้งานบน Pi

- [ ] โค้ดอยู่ที่ `/opt/topgun` และไฟล์ `setup.sh` อยู่ที่ `/opt/topgun/setup.sh`
- [ ] มี system user `topgun` (`id topgun` ต้องพบผู้ใช้)
- [ ] มี `/etc/topgun/topgun.env` ที่ systemd อ่านได้ และตั้ง `ROASTML_MODEL`, `TOPGUN_PORT` รวมถึง HiveMQ credentials แล้ว; จำกัดสิทธิ์ไฟล์และอย่า commit password
- [ ] ติดตั้ง systemd units จาก `Firmware/deploy/` ไปที่ `/etc/systemd/system/`: `coffee_service.service`, `coffee_subscriber.service` และ `topgun-boot-check.service`
- [ ] รัน `sudo systemctl daemon-reload` หลังติดตั้งหรือแก้ unit files
- [ ] Enable และ start เว็บกับ MQTT subscriber: `sudo systemctl enable --now coffee_service.service coffee_subscriber.service`
- [ ] Enable boot check: `sudo systemctl enable topgun-boot-check.service`; unit นี้เรียก `/opt/topgun/Firmware/deploy/boot-check.sh` หลัง `coffee_service.service` เริ่มทำงาน
- [ ] ตรวจว่าเว็บและ subscriber เป็น `active` ด้วย `systemctl is-active coffee_service.service coffee_subscriber.service`
- [ ] ตรวจ subscriber ด้วย `sudo journalctl -u coffee_subscriber.service -n 20 --no-pager`; ควรเห็น `subscribed to coffee/roast/result`
- [ ] ตรวจผล boot check ด้วย `sudo journalctl -u topgun-boot-check.service -b --no-pager`; ต้องเห็น `Topgun health check passed`
- [ ] ยืนยันว่า env file มี port เดียวกับ URL ที่ใช้งาน และ health check ใช้ port เดียวกัน (ค่าเริ่มต้น `8080`)
- [ ] ถ้าใช้ hotspot ให้ตั้ง NetworkManager ตาม `Firmware/deploy/hotspot.sh`; ใน repo ไม่มี systemd unit สำหรับ hotspot
- [ ] ก่อนรัน `./setup.sh` ตรวจ Python และ architecture บน Pi; wheelhouse ต้องตรงกับค่าจริง ไม่ใช่แค่ชื่อรุ่น Pi
- [ ] Pi ที่ตรวจครั้งล่าสุดคือ Raspberry Pi 5 / Debian 13 (Trixie) / Python 3.13.5 / `aarch64`; ตรวจ `python3 --version` และ architecture ซ้ำก่อนจัด wheelhouse ถ้าเปลี่ยน Python เป็น 3.11 ให้ใช้ NumPy 2.4.6 ตาม marker ใน requirements และเตรียม wheel `cp311` ให้ตรง architecture

## โครงสร้างโปรเจกต์

```text
Topgun/
├── Firmware/                     # เพื่อน
│   ├── app.py                    # /, POST /api/predict, /market, /health
│   ├── mqtt_pub.py               # queue → MQTT (ไม่อยู่ใน request path)
│   ├── service.py                # subscribe → validate → SQLite (UNIQUE msg_id)
│   ├── templates/                # index.html, market.html
│   ├── static/                   # script.js (ย่อรูป + performance.now) — ไม่ใช้ CDN
│   ├── deploy/                   # systemd services, hotspot.sh, boot-check.sh
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
