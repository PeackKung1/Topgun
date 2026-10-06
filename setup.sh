#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WHEELHOUSE="${WHEELHOUSE:-$ROOT/wheelhouse}"
VENV="${VENV:-$ROOT/.venv}"

if [[ ! -d "$WHEELHOUSE" ]]; then
  echo "Offline wheelhouse not found: $WHEELHOUSE" >&2
  echo "Put the matching Python/CPU wheels for ML/requirements-pi.txt and Firmware/requirements-fw.txt there." >&2
  exit 2
fi

python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "roastml requires Python >= 3.11; found $python_version" >&2
  exit 2
fi

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" \
  -r "$ROOT/ML/requirements-pi.txt" \
  -r "$ROOT/Firmware/requirements-fw.txt"
"$VENV/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" -e "$ROOT/ML" --no-deps

echo "Offline Python install complete in $VENV"
echo "Next: review Firmware/deploy and configure the service account, env file, broker, and NetworkManager hotspot for this Pi."
