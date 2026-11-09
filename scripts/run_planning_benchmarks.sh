#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for run in 1 2 3; do
    ./scripts/run_npu.sh scripts/benchmark_tree_grid.py --seed "$((200+run))" --output "results/tree_grid_run${run}.json" > "results/tree_grid_run${run}.log" 2>&1
    ./scripts/run_npu.sh scripts/benchmark_decode_plan.py --seed "$((300+run))" --output "results/decode_plan_run${run}.json" > "results/decode_plan_run${run}.log" 2>&1
done
.venv/bin/python scripts/analyze_planning.py tree results/tree_grid_run{1,2,3}.json --output results/tree_grid.json
.venv/bin/python scripts/analyze_planning.py decode results/decode_plan_run{1,2,3}.json --output results/decode_plan.json
