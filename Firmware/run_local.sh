#!/usr/bin/env bash
set -euo pipefail

FIRMWARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$FIRMWARE_DIR")"
PYTHON_BIN="${TOPGUN_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
LOCAL_HOST="${TOPGUN_LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${TOPGUN_PORT:-8080}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Create it with python3 -m venv .venv and install Firmware/requirements-fw.txt, Pillow, and ML/roastml." >&2
  exit 2
fi

read -r -p "HiveMQ username: " TOPGUN_MQTT_USERNAME
read -r -s -p "HiveMQ password (hidden): " TOPGUN_MQTT_PASSWORD
printf '\n'
export TOPGUN_MQTT_USERNAME TOPGUN_MQTT_PASSWORD
export TOPGUN_MQTT_HOST="topguncoffee-a863a362.a02.usw2.aws.hivemq.cloud"
export TOPGUN_MQTT_PORT=8883
export TOPGUN_MQTT_TOPIC="coffee/roast/result"
export TOPGUN_MQTT_TLS=1
export TOPGUN_MQTT_ENABLED=1
export ROASTML_MODEL=stub
export TOPGUN_DB_PATH="${TMPDIR:-/tmp}/topgun-local-check.sqlite3"
export LOG_LEVEL=INFO
export TOPGUN_LOCAL_HOST="$LOCAL_HOST"

cd "$FIRMWARE_DIR"
"$PYTHON_BIN" -m service &
SUBSCRIBER_PID=$!
trap 'kill "$SUBSCRIBER_PID" 2>/dev/null || true' EXIT INT TERM

if [[ "$LOCAL_HOST" == "0.0.0.0" ]]; then
  echo "Starting dashboard on all computer network interfaces at port $LOCAL_PORT"
  echo "On this computer: http://127.0.0.1:$LOCAL_PORT"
  echo "From a phone on the same Wi-Fi: http://<computer-wifi-ip>:$LOCAL_PORT"
else
  echo "Starting dashboard at http://127.0.0.1:$LOCAL_PORT (computer only)"
fi
echo "Submit one image to publish a stub result to HiveMQ; press Ctrl+C to stop."
"$PYTHON_BIN" -c 'import logging, os; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"); from waitress import serve; from app import app; serve(app, host=os.getenv("TOPGUN_LOCAL_HOST", "127.0.0.1"), port=int(os.getenv("TOPGUN_PORT", "8080")), threads=4)'
