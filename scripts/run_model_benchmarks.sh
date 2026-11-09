#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_path="$1"
for planning in rebuild cached; do
    for run in 1 2 3; do
        ./scripts/run_npu.sh scripts/benchmark_qwen.py --model "$model_path" --planning "$planning" --seed "$((400+run))" --output "results/qwen4b_${planning}_run${run}.json" > "results/qwen4b_${planning}_run${run}.log" 2>&1
    done
done
.venv/bin/python scripts/analyze_model.py results/qwen4b_{rebuild,cached}_run{1,2,3}.json --output results/qwen4b.json
