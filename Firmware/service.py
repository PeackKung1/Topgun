"""MQTT subscriber, event validation, and SQLite persistence."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

LOG = logging.getLogger("topgun.service")
LABELS = {"light", "medium", "dark"}
STATUSES = {"ok", "low_confidence", "bad_image", "error"}
RESULT_KEYS = {
    "status", "label", "label_th", "message_th", "confidence", "probs", "warnings",
    "n_beans", "proportions", "beans", "timing_ms", "model",
}


def database_path() -> Path:
    return Path(os.getenv("TOPGUN_DB_PATH", str(Path(__file__).resolve().parent / "data" / "predictions.sqlite3")))


def connect_db(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        msg_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        label TEXT,
        confidence REAL,
        model TEXT NOT NULL,
        result_json TEXT NOT NULL
    )""")
    db.commit()
    return db


def validate_event(payload: Any) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("event must be an object")
    msg_id, created_at, result = payload.get("msg_id"), payload.get("created_at"), payload.get("result")
    try:
        uuid.UUID(str(msg_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid msg_id") from exc
    if not isinstance(created_at, str) or len(created_at) > 40:
        raise ValueError("invalid created_at")
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        raise ValueError("result fields do not match roastml contract")
    if result["status"] not in STATUSES:
        raise ValueError("invalid status")
    if result["status"] in {"ok", "low_confidence"} and result["label"] not in LABELS:
        raise ValueError("invalid label")
    if result["status"] in {"bad_image", "error"} and result["label"] is not None:
        raise ValueError("failure result must not have a label")
    if not isinstance(result["message_th"], str) or not isinstance(result["model"], str):
        raise ValueError("invalid message/model")
    return str(msg_id), created_at, result


def persist_event(payload: Any, path: str | os.PathLike[str] | None = None) -> bool:
    msg_id, created_at, result = validate_event(payload)
    with connect_db(path) as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO predictions(msg_id, created_at, status, label, confidence, model, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, created_at, result["status"], result["label"], result["confidence"], result["model"],
             json.dumps(result, ensure_ascii=False, allow_nan=False)),
        )
        return cursor.rowcount == 1


def read_market_snapshot(path: str | os.PathLike[str] | None = None, limit: int = 50) -> dict[str, Any]:
    with connect_db(path) as db:
        counts = {row["label"]: row["n"] for row in db.execute(
            "SELECT label, COUNT(*) AS n FROM predictions WHERE status IN ('ok','low_confidence') "
            "AND label IS NOT NULL GROUP BY label"
        )}
        rows = db.execute(
            "SELECT created_at, status, label, confidence, model FROM predictions "
            "ORDER BY created_at DESC LIMIT ?", (limit,),
        ).fetchall()
        total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {"counts": {label: counts.get(label, 0) for label in sorted(LABELS)}, "total": total,
            "recent": [dict(row) for row in rows]}


def run_subscriber() -> None:
    import paho.mqtt.client as mqtt

    host = os.getenv("TOPGUN_MQTT_HOST", "topguncoffee-a863a362.a02.usw2.aws.hivemq.cloud")
    port = int(os.getenv("TOPGUN_MQTT_PORT", "8883"))
    topic = os.getenv("TOPGUN_MQTT_TOPIC", "coffee/roast/result")
    client_id = os.getenv("TOPGUN_MQTT_SERVICE_CLIENT_ID", "topgun-sqlite-service")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    username, password = os.getenv("TOPGUN_MQTT_USERNAME"), os.getenv("TOPGUN_MQTT_PASSWORD")
    if username:
        client.username_pw_set(username, password)
    if os.getenv("TOPGUN_MQTT_TLS", "1").lower() not in {"0", "false", "no"}:
        client.tls_set()

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code == 0:
            client.subscribe(topic, qos=1)
            LOG.info("subscribed to %s", topic)
        else:
            LOG.error("MQTT connection rejected: %s", reason_code)

    def on_message(_client, _userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            inserted = persist_event(payload)
            LOG.info("%s event msg_id=%s", "stored" if inserted else "duplicate", payload.get("msg_id"))
        except Exception:
            LOG.exception("invalid MQTT event; discarded")

    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(host, port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_subscriber()
