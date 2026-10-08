#!/usr/bin/env bash
set -euo pipefail

health_url="${TOPGUN_HEALTH_URL:-http://127.0.0.1:${TOPGUN_PORT:-8080}/health}"
python_bin="${TOPGUN_PYTHON:-/opt/topgun/.venv/bin/python}"
for attempt in $(seq 1 30); do
  if "$python_bin" -c 'import json, sys, urllib.request; r=urllib.request.urlopen(sys.argv[1], timeout=2); body=json.load(r); model=body.get("ml", {}).get("model"); raise SystemExit(0 if r.status == 200 and body.get("status") == "ok" and isinstance(model, str) and model != "stub" else 1)' "$health_url"; then
    echo "Topgun health check passed: $health_url"
    exit 0
  fi
  sleep 1
done
echo "Topgun health check failed after 30 attempts: $health_url" >&2
exit 1
