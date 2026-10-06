# Topgun

โครงสร้างโปรเจกต์ Topgun แบ่งเป็นส่วน Firmware และ ML ดังนี้

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
