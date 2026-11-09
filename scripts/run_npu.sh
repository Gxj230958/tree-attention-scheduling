#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE:-14}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
status=$(npu-smi info)
printf '%s\n' "$status"
NPU_STATUS="$status" .venv/bin/python - "$@" <<'PY'
import os, re, sys
status = os.environ.pop('NPU_STATUS')
card = os.environ['ASCEND_RT_VISIBLE_DEVICES']
row = re.search(r'\|\s+'+card+r'\s+910B2C\s+\|\s+OK\b[^\n]*\n\|[^\n]*\|\s+0\s+', status)
idle = re.search(r'No running processes found in NPU '+card+r'\s+\|', status) is not None
if not row or not idle:
    raise SystemExit(f'NPU {card} must be healthy and idle')
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
PY
