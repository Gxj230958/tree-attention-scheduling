# Tree-Shared Attention Scheduling for Ascend Cube

Cost-model-guided grouping for tree-shared LLM decode attention, implemented as Triton Ascend matrix kernels on a single Ascend 910B NPU and integrated end-to-end with pretrained Qwen3-4B.

## Overview

Requests that share prompt prefixes are stored once in a compressed radix tree; a scheduling policy decides which requests compute each shared context segment together, and a hardware-aware cost model prices every candidate grouping before execution.

- **Subset dynamic programming with full descendant cost.** Extends FastTree-style (MLSys'25) cost-model-guided grouping: at each tree node the DP selects a subset of child groups to absorb the inherited context, minimizing a recurrence that accounts complete descendant subtree cost rather than evaluating children locally. Baselines: per-request execution, fixed tree-node sharing, and the per-child SplitQ/SplitK greedy heuristic.
- **Hardware-aware cost terms.** The cost function models Cube query-row padding (each private query contributes 4 grouped-query rows padded to a 16-row tile), KV reads split at physical discontinuities, matrix work, temporary partial outputs, and the global log-sum-exp merge of unequal segments. Matrix tiles (16×128 for small groups, 64×64 for large) come from an independent calibration grid disjoint from all evaluation shapes.
- **Measured results.** On 128 requests with 8192-token contexts sharing 8128 tokens (16,320 physical vs 1,048,576 logical KV positions), resident attention takes 2.523 ms versus 2.839 ms for the faster vendor fused-operator layout (1.126×); across 81 held-out tree shapes the geometric-mean resident speedup over per-request execution is 1.315×. Reusing the prompt plan cuts a 32-request, 16-step decode workload's planning-plus-attention time from 3.896 ms to 0.606 ms per step (fixed cached grouping: 0.536 ms).
- **Pretrained model validation.** Qwen3-4B runs on the shared-KV tree with reusable decode plans: 1,224 same-Q/K/V layer comparisons pass against independent FP32 attention and the vendor operator (maximum layer relative L2 error 0.218%, vendor 0.245%); cached tree attention reduces document-QA decode time by 3.3% relative to per-request execution, with 100% top-token agreement on three of four public-text workloads.
- **Honest scope.** The vendor resident operator is faster than every tree strategy on five of six representative shapes with limited sharing; the DP matches fixed grouping's aggregate resident latency (1.008×) at higher planning cost. All conclusions are single-device.

## Hardware

All measurements use one Ascend 910B2C NPU (CANN 8.5.1, Triton Ascend, BF16, 32 query heads / 8 KV heads / head dim 128) targeting the Cube matrix units. The scheduling policy is portable: only the analytical cost parameters and tile calibration are hardware-specific, not the tree/DP machinery.

## Links

- Technical report (methods, protocol, full measurements, limitations): [`reports/technical_report.md`](reports/technical_report.md)
- FastTree paper (MLSys 2025), the cost-model line this work extends: https://panzaifeng.github.io/assets/pdf/mlsys25fasttree.pdf
- FastTree reference artifact (pinned commit `167928b`): https://github.com/PanZaifeng/FastTree-Artifact
- Model and data sources: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

## Repository contents

- `src/tree_attention/` — compressed radix tree (`tree.py`), schedulers and cost model (`schedule.py`), Triton Ascend Cube kernels and merge (`kernels.py`), FP32 reference (`reference.py`), Qwen3 adapter with shared KV pools and cached plans (`qwen.py`).
- `scripts/` — correctness checks, tile calibration, representative-shape and 81-tree grid benchmarks, decode-plan-reuse experiment, Qwen3-4B generation benchmarks, aggregation and PDF plots.
- `tests/` — CPU semantics tests: unequal nested prefixes, internal termination, duplicates, deep trees, exact path partitioning, planner-state lifetime.
- `configs/` — recorded environment, model file revisions, exact public-text workload token IDs.
- `results/` — raw per-process JSON records, aggregates, and four PDF figures behind every table in the report.
- `data/` — small derived SQuAD 2.0 excerpt used to build the four workloads.
- `reports/` — English technical report, Chinese explanation, application/CV summary, tutorial Q&A.

## Reproduction

Requires an Ascend 910B with CANN (recorded environment in `configs/environment.json`). Install inside the project:

```bash
/usr/local/python3.11.14/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps -e .
.venv/bin/pip install matplotlib
OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
```

`scripts/run_npu.sh` loads CANN and verifies the selected physical card (`NPU_DEVICE`, 14 in the recorded runs) is healthy and idle before launch:

```bash
NPU_DEVICE=14 ./scripts/run_npu.sh scripts/check_npu.py
NPU_DEVICE=14 ./scripts/run_npu.sh scripts/check_qwen.py --planning cached --output results/qwen_cached_correctness.json
```

Three independent processes with fixed seeds for the kernel benchmarks (coordinate an exclusive performance window first):

```bash
NPU_DEVICE=14 ./scripts/run_benchmarks.sh           # 6 representative shapes + 81-tree grid
NPU_DEVICE=14 ./scripts/run_planning_benchmarks.sh  # 16-step decode plan reuse
```

Pretrained Qwen3-4B (supply a local copy of the official weights and tokenizer):

```bash
.venv/bin/python scripts/make_workloads.py --model /path/to/Qwen3-4B
./scripts/run_model_benchmarks.sh /path/to/Qwen3-4B  # both planning modes, seeds 401–403
```

`scripts/check_divergence.py --model /path/to/Qwen3-4B --input results/qwen4b_cached_run1.json` locates the first multi-branch generation difference and reports top-two logit margins. Full command details are in the technical report; exact tokens for all workloads are in `configs/qwen_workloads.json`.

## Limitations

Single-device, single-token decode, full attention, fixed batch, Qwen3 grouped-query attention; prefill uses the original model path. Cost-model roofline constants are analytical estimates with independently calibrated tiles. Four short public-text generation batches establish integration and timing, not broad model quality. BF16 paths can diverge at near-tied logits (measured multi-branch token agreement 89.84%); synthetic peak-memory figures include resident reference layouts.
