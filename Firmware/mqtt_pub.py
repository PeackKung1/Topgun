"""Non-blocking in-process queue and background MQTT publisher."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

LOG = logging.getLogger("topgun.mqtt_pub")
_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=int(os.getenv("TOPGUN_MQTT_QUEUE_SIZE", "500")))
_START_LOCK = threading.Lock()
_STARTED = False


def enqueue_prediction(result: dict[str, Any]) -> bool:
    """Queue an event without waiting for MQTT or SQLite; return False if full."""
    event = {
        "msg_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "result": result,
    }
    try:
        _QUEUE.put_nowait(event)
        return True
    except queue.Full:
        LOG.error("prediction queue full; dropping event msg_id=%s", event["msg_id"])
        return False


def start_publisher() -> None:
    global _STARTED
    with _START_LOCK:
        if _STARTED:
            return
        thread = threading.Thread(target=_publish_loop, name="mqtt-publisher", daemon=True)
        thread.start()
        _STARTED = True


def _publish_loop() -> None:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        LOG.exception("paho-mqtt is missing; prediction events will not be published")
        return

    host = os.getenv("TOPGUN_MQTT_HOST", "topguncoffee-a863a362.a02.usw2.aws.hivemq.cloud")
    port = int(os.getenv("TOPGUN_MQTT_PORT", "8883"))
    topic = os.getenv("TOPGUN_MQTT_TOPIC", "coffee/roast/result")
    client_id = os.getenv("TOPGUN_MQTT_CLIENT_ID", f"topgun-publisher-{uuid.uuid4().hex[:8]}")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    username, password = os.getenv("TOPGUN_MQTT_USERNAME"), os.getenv("TOPGUN_MQTT_PASSWORD")
    if username:
        client.username_pw_set(username, password)
    if os.getenv("TOPGUN_MQTT_TLS", "1").lower() not in {"0", "false", "no"}:
        client.tls_set()

    connected = threading.Event()

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        if reason_code == 0:
            connected.set()
            LOG.info("connected to MQTT broker %s:%s", host, port)
        else:
            connected.clear()
            LOG.error("MQTT connection rejected: %s", reason_code)

    def on_disconnect(_client, _userdata, _disconnect_flags, reason_code, _properties):
        connected.clear()
        LOG.warning("disconnected from MQTT broker: %s", reason_code)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    while True:
        try:
            client.connect(host, port, keepalive=30)
            client.loop_start()
            break
        except OSError:
            LOG.exception("MQTT broker unavailable at %s:%s; retrying", host, port)
            time.sleep(5)

    while True:
        event = _QUEUE.get()
        try:
            # Keep the HTTP request independent from broker availability. The worker
            # may wait here while the broker reconnects; bounded queue limits RAM use.
            while not connected.wait(timeout=1):
                pass
            info = client.publish(topic, json.dumps(event, ensure_ascii=False), qos=1)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                LOG.error("MQTT publish failed rc=%s msg_id=%s", info.rc, event["msg_id"])
        except Exception:
            LOG.exception("MQTT publish failed msg_id=%s", event.get("msg_id"))
        finally:
            _QUEUE.task_done()
