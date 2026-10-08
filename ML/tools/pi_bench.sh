#!/usr/bin/env bash
# Run ON the Raspberry Pi (human-run). Records hardware/runtime, runs the
# torch/sklearn-free tests, then benchmarks bytes->dict for each model dir.
#   ML_DIR=/opt/topgun/ML DATA_DIR=/opt/topgun-data PY=/opt/topgun/.venv/bin/python \
#     bash ML/tools/pi_bench.sh /opt/topgun/ML/models/current /opt/topgun/ML/models/<candidate> ...
# Trainval images only (bench_candidates refuses split=test). Outputs are new files.
set -euo pipefail
ML_DIR="${ML_DIR:-/opt/topgun/ML}"
DATA_DIR="${DATA_DIR:-/opt/topgun-data}"
PY="${PY:-/opt/topgun/.venv/bin/python}"
OUT="$ML_DIR/results/pi_$(date +%Y%m%dT%H%M%S)"
[ "$#" -ge 1 ] || { echo "usage: $0 <model_dir> [model_dir ...]" >&2; exit 2; }
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"
  echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  echo "uname: $(uname -a)"
  grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true
  grep MemTotal /proc/meminfo || true
  echo "python: $("$PY" -c 'import sys,platform; print(sys.version.split()[0], platform.machine())')"
  vcgencmd measure_temp 2>/dev/null || true
  vcgencmd get_throttled 2>/dev/null || true
} | tee "$OUT/hardware.txt"
"$PY" -m pip freeze --all > "$OUT/pip_freeze.txt"
"$PY" -c 'import onnxruntime as o, numpy, cv2, PIL; print("ort", o.__version__, o.get_available_providers()); print("numpy", numpy.__version__, "cv2", cv2.__version__, "pillow", PIL.__version__)' | tee "$OUT/runtime.txt"
(cd "$ML_DIR" && PYTHONPATH="$ML_DIR" "$PY" -B -m pytest tests/test_contract.py tests/test_linear_backend.py tests/test_onnx_backend.py \
   -q -p no:cacheprovider) 2>&1 | tail -5 | tee "$OUT/pytest.txt" || echo "pytest failed or missing (see pytest.txt); continuing to bench"
for model in "$@"; do
  name="$(basename "$model")"
  PYTHONPATH="$ML_DIR" "$PY" -B -m tools.bench_candidates --model "$model" --data-dir "$DATA_DIR" \
    --output "$OUT/bench_${name}.json" --host pi --per-source 50 --seed 20261006
done
echo "results in $OUT"
