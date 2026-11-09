#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
for run in 1 2 3; do
    ./scripts/run_npu.sh scripts/benchmark.py --repeats 1 --run-id "$run" --seed "$((100+run))" --output "results/benchmark_run${run}.json" > "results/benchmark_run${run}.log" 2>&1
done
.venv/bin/python scripts/aggregate_benchmarks.py results/benchmark_run1.json results/benchmark_run2.json results/benchmark_run3.json
.venv/bin/python scripts/plot_results.py
