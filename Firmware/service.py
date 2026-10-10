"""MQTT subscriber, event validation, and SQLite persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
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
V3_KEYS = RESULT_KEYS | {"schema_version", "counts", "count_method", "image_size"}
_DB_INIT_LOCK = threading.Lock()


def database_path() -> Path:
    return Path(os.getenv("TOPGUN_DB_PATH", str(Path(__file__).resolve().parent / "data" / "predictions.sqlite3")))


def connect_db(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=5)
    db.row_factory = sqlite3.Row
    # WAL setup does not consistently honor busy_timeout on a brand-new DB.
    # Serialize threads and retry briefly for a simultaneous subscriber process.
    deadline = time.monotonic() + 5
    with _DB_INIT_LOCK:
        while True:
            try:
                _initialize_db(db)
                return db
            except sqlite3.OperationalError as exc:
                db.rollback()
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    db.close()
                    raise
                time.sleep(.01)


def _initialize_db(db: sqlite3.Connection) -> None:
    if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
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
    # Serialize first-start migrations across concurrent web/subscriber calls.
    db.execute("BEGIN IMMEDIATE")
    if "estimated" not in {r[1] for r in db.execute("PRAGMA table_info(predictions)")}:
        db.execute("ALTER TABLE predictions ADD COLUMN estimated INTEGER NOT NULL DEFAULT 0")
    db.commit()


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
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    v3 = result.get("schema_version") == 3
    allowed = (V3_KEYS, V3_KEYS - {"beans"}) if v3 else (RESULT_KEYS,)
    if set(result) not in allowed:
        raise ValueError("result fields do not match roastml contract")
    if result["status"] not in STATUSES:
        raise ValueError("invalid status")
    if result["status"] in {"ok", "low_confidence"} and result["label"] not in LABELS:
        raise ValueError("invalid label")
    if result["status"] in {"bad_image", "error"} and result["label"] is not None:
        raise ValueError("failure result must not have a label")
    if not isinstance(result["message_th"], str) or not isinstance(result["model"], str):
        raise ValueError("invalid message/model")
    if v3:
        _validate_v3(result)
    return str(msg_id), created_at, result


def _validated_counts(result: dict) -> dict[str, int] | None:
    """Counts present means authoritative: never reconstruct from proportions."""
    counts, n = result.get("counts"), result.get("n_beans")
    if counts is None and n is None:
        return None
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("invalid n_beans")
    if not isinstance(counts, dict) or set(counts) != LABELS or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()
    ) or sum(counts.values()) != n:
        raise ValueError("counts must sum to n_beans")
    return counts


def _is_estimated(result: dict) -> bool:
    return result.get("count_method") == "estimated" or "bean_count_estimated" in (result.get("warnings") or [])


def _validate_v3(result: dict) -> None:
    if type(result["schema_version"]) is not int:
        raise ValueError("invalid schema_version")
    counts = _validated_counts(result)
    warnings = result["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError("invalid warnings")
    size = result["image_size"]
    if size is not None and (not isinstance(size, list) or len(size) != 2 or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in size
    )):
        raise ValueError("invalid image_size")
    if counts is None:
        if result["count_method"] is not None or result.get("beans"):
            raise ValueError("unavailable counting must be null")
        return
    method = result["count_method"]
    if method not in {"exact", "estimated"} or (method == "estimated") != ("bean_count_estimated" in warnings):
        raise ValueError("count_method/warning mismatch")
    n = result["n_beans"]
    if result["status"] not in {"ok", "low_confidence"}:
        raise ValueError("failure result cannot contain counts")
    if n and counts[result["label"]] != max(counts.values()):
        raise ValueError("label must be a count majority")
    if (not n and "no_beans_detected" not in warnings) or (
        sum(v > 0 for v in counts.values()) > 1 and "mixed_roast" not in warnings
    ):
        raise ValueError("missing count warning")
    if "beans" in result:
        beans = result["beans"]
        if not isinstance(beans, list) or len(beans) != n:
            raise ValueError("beans length must equal n_beans")
        if any(not isinstance(b, dict) or b.get("label") not in LABELS for b in beans):
            raise ValueError("invalid bean labels")
        if counts != {c: sum(b["label"] == c for b in beans) for c in LABELS}:
            raise ValueError("bean labels disagree with counts")


def persist_event(payload: Any, path: str | os.PathLike[str] | None = None) -> bool:
    msg_id, created_at, result = validate_event(payload)
    with connect_db(path) as db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO predictions(msg_id, created_at, status, label, confidence, model, result_json, estimated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, created_at, result["status"], result["label"], result["confidence"], result["model"],
             json.dumps(result, ensure_ascii=False, allow_nan=False), int(_is_estimated(result))),
        )
        return cursor.rowcount == 1


def read_market_snapshot(path: str | os.PathLike[str] | None = None, limit: int = 50) -> dict[str, Any]:
    with connect_db(path) as db:
        rows = db.execute(
            "SELECT created_at, status, result_json FROM predictions "
            "WHERE status IN ('ok','low_confidence') ORDER BY created_at DESC"
        ).fetchall()

    counts = {label: 0 for label in LABELS}
    recent_beans: list[dict[str, Any]] = []
    bean_total = 0
    estimated_results = 0
    visible_only_results = 0
    recent_results: list[dict[str, Any]] = []
    for row in rows:
        try:
            result = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue

        beans = result.get("beans")
        valid_beans = [
            bean for bean in beans
            if isinstance(bean, dict) and bean.get("label") in LABELS
        ] if isinstance(beans, list) else []

        # Prefer per-bean detections when they account for the full aggregate.
        # If only some detections are present, fall back to n_beans/proportions
        # so the market totals do not silently drop the unlisted beans.
        n_beans = result.get("n_beans")
        proportions = result.get("proportions")
        aggregate_counts = None
        if "counts" not in result and isinstance(n_beans, int) and not isinstance(n_beans, bool) and n_beans > 0:
            if isinstance(proportions, dict) and all(
                isinstance(proportions.get(label), (int, float)) and not isinstance(proportions.get(label), bool)
                for label in LABELS
            ):
                values = {label: float(proportions[label]) for label in LABELS}
                if all(math.isfinite(value) and value >= 0 for value in values.values()):
                    proportion_total = sum(values.values())
                    if proportion_total > 0:
                        raw = {label: n_beans * values[label] / proportion_total for label in LABELS}
                        aggregate_counts = {label: int(raw[label]) for label in LABELS}
                        remainder = n_beans - sum(aggregate_counts.values())
                        for label in sorted(
                            LABELS,
                            key=lambda item: raw[item] - aggregate_counts[item],
                            reverse=True,
                        )[:remainder]:
                            aggregate_counts[label] += 1

        if "counts" in result:
            # Includes zero and null: neither may fall back to rounded proportions.
            try:
                per_result_counts = _validated_counts(result)
            except ValueError:
                continue
            if per_result_counts is None:
                continue
        elif valid_beans and (aggregate_counts is None or len(valid_beans) == n_beans):
            per_result_counts = {label: 0 for label in LABELS}
            for bean in valid_beans:
                per_result_counts[bean["label"]] += 1
        elif aggregate_counts is not None:
            per_result_counts = aggregate_counts
        elif valid_beans:
            per_result_counts = {label: 0 for label in LABELS}
            for bean in valid_beans:
                per_result_counts[bean["label"]] += 1
        else:
            continue

        estimated = _is_estimated(result)
        visible_only = "count_visible_only" in (result.get("warnings") or [])
        if estimated:
            estimated_results += 1
        visible_only_results += int(visible_only)
        for label, count in per_result_counts.items():
            counts[label] += count
            bean_total += count
        if len(recent_results) < limit:
            recent_results.append({"created_at": row["created_at"], "n_beans": sum(per_result_counts.values()),
                                   "counts": per_result_counts, "estimated": estimated, "count_visible_only": visible_only})

        if valid_beans:
            for index, bean in enumerate(valid_beans, start=1):
                recent_beans.append({
                    "created_at": row["created_at"],
                    "index": index,
                    "label": bean["label"],
                    "confidence": bean.get("conf"),
                })

    return {
        "counts": counts,
        "bean_total": bean_total,
        # Keep `total` for existing clients; it now means total beans.
        "total": bean_total,
        "estimated_results": estimated_results,
        "visible_only_results": visible_only_results,
        "recent_results": recent_results,
        "recent_beans": recent_beans[:limit],
    }


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
