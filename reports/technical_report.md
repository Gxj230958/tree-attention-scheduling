# Tree-shared attention on Ascend 910B

Shared-prefix matrix attention processes 128 requests with 8192-token contexts in 2.523 ms on Ascend 910B, versus 2.839 ms for the faster vendor layout. Across 81 held-out tree shapes, it achieves a 1.315× geometric-mean speedup over per-request tree execution. Reusing the prompt plan reduces a 32-request decode workload's planning-plus-attention time from 3.896 ms to 0.606 ms per step; fixed grouping with a cached plan reaches 0.536 ms.

## Representation and scheduling

A compressed radix tree stores exact token-prefix sharing. Each node records its KV positions, descendant requests, and requests ending at that node. Duplicate contexts, unequal branch lengths, and internal terminal nodes are supported. Every generated schedule partitions each request's context exactly once.

The four methods are per-request execution, fixed tree-node sharing, a per-child FastTree-style SplitQ/SplitK heuristic, and a subset dynamic program. Let `P` be inherited context, `K_v` the current node's context, and `R_v` its requests. The dynamic program selects a subset `S` of child groups to receive `P + K_v`. Remaining requests share one computation of that context. Its recurrence is:

```
T(v,P) = min_S [ C(|R_v| - sum_{c in S}|R_c|, P + K_v)
                + sum_{c in S} T(c, P + K_v)
                + sum_{c not in S} T(c, empty) ]
```

A child-by-child table indexed by the removed request count evaluates this choice. Requests terminating at the current node remain in its group. The local heuristic evaluates each child's immediate SplitQ/SplitK cost without the dynamic program's full descendant cost.

`C` includes Cube row padding, KV reads, matrix work, temporary outputs, and merging. It first applies query tiling, context splitting, and splitting at physical KV discontinuities. Contiguous physical reads are therefore part of both the plan and its cost. Planning state is released when the call returns.

Independent calibration on 24-request, 768-token contexts chooses 16 query rows × 128 keys for small groups and 64 rows × 64 keys for larger groups. Separate launches serve those phases. Context segments contain at most 512 tokens, and query groups contain at most 64 rows after grouped-query head expansion.

## Kernel and model integration

K is resident in `[KV head, head dimension, token]` order; V uses `[KV head, token, head dimension]`. Query heads sharing a KV head form adjacent matrix rows. Triton Ascend matrix products compute QK and weighted V. Each segment maintains its running maximum, denominator, and value accumulator in FP32 and writes its normalized result and log-sum-exp in contiguous query-row order.

For partial outputs `o_s` and log-sum-exp values `l_s`, global normalization is:

```
o = sum_s exp(l_s - max_t l_t) * o_s / sum_s exp(l_s - max_t l_t)
```

The merge uses each request's segment list, preserving full-context attention across unequal segments.

The Qwen3 adapter constructs one shared physical KV pool per layer during causal prefill. It preserves projections, Q/K normalization, rotary positions, output projections, MLPs, and vocabulary projection. The rebuilding path updates the compressed private leaves and plans the full tree each step. Both paths reserve one contiguous generated-token tail per request. The cached path retains the prompt plan. Each decode step writes the new KV values and changes tail lengths; the plan and scratch buffers are shared across model layers.

## Correctness

Seven CPU tests cover unequal nested prefixes, internal termination, duplicate contexts, no sharing, deep trees, extreme logits, exact path partitioning, private-path extension, estimated-cost ordering, and planner-state lifetime. The independent reference performs full attention separately for each request in FP32.

The NPU suite passes 32 comparisons across four trees, two precisions, and four schedulers. Maximum absolute error is 0.0009765625 in FP16 and 0.0078125 in BF16. All 81 evaluation trees also pass the independent attention reference for all four strategies in each of three processes: 972 comparisons.

The two-layer random-weight Qwen3 model passes 12 full-sequence logit comparisons for each of rebuilding and cached execution. Maximum logit error is 0.0005493164 and top-token agreement is 100%. The 16-step plan experiment checks complete attention at the first and last steps for every method and execution mode.

The public Qwen3-4B checkpoint passes 1,224 layer-level comparisons across rebuilding, cached execution, the shared-document workload, and common-history generation divergence. Each comparison supplies the identical Q/K/V values and context to the tree kernel, the vendor fused operator, and an independent FP32 full-attention reference. The maximum layer relative L2 error is 0.218% for the tree kernel and 0.245% for the vendor operator. The maximum absolute error across these checks is 0.125 for both kernels.

At the full-model output, tree versus full-sequence BF16 evaluation has maximum relative L2 error 2.692% and maximum absolute logit difference 0.5. Standard Hugging Face cached evaluation versus its own full-sequence evaluation reaches 2.868% and 0.75. All top-token comparisons in the short-context and shared-document checks agree. Different batching and reduction paths can produce different floating-point outputs despite identical mathematical operations; the same-Q/K/V layer reference isolates attention correctness from accumulated model-path differences ([PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)). Raw layer and logit measurements are in `results/qwen4b_{cached,rebuild,document}_correctness.json`.

## Measurement protocol

Experiments use one Ascend 910B2C, 32 query heads, eight KV heads, dimension 128, and BF16. Formal runs use exclusive performance windows. Three separate processes use seeds 101–103 for representative shapes, 201–203 for the tree grid, and 301–303 for growing decode tails. Compilation and warmup precede resident execution timing. Raw files retain independent samples, metadata construction, planning, layout conversion, numerical error, and first-call costs.

The vendor baseline is `torch_npu.npu_fused_infer_attention_score` with BNSD and BSH layouts and actual sequence lengths. It starts from the same shared KV pool; materializing its padded per-request layout is measured separately. Resident attention reuses that layout. Cold preparation counts the conversion once. Each tree-kernel invocation includes its query-layout conversion.

The evaluation grid contains every combination of branch fanouts `(3,1)`, `(9,3)`, `(17,7)`; root lengths 32, 192, 768; shared-child scales 12, 96, 384; and private-length scales 24, 192, 768. Child and private lengths differ between branches. Storage follows depth-first trie order. All 81 shapes are retained. The calibration grid uses separate lengths and a different four-request topology.

## Representative attention latency

Resident latency is in microseconds, shown as median [minimum, maximum] across three independent processes. The final pair has exactly 128 requests and 8192 context tokens per request. The shared member contains an 8128-token root and 64 private tokens per request; its physical pool contains 16,320 tokens, compared with 1,048,576 logical tokens.

| Shape | Per request | Fixed | Greedy | Dynamic | Vendor BNSD | Vendor BSH |
|---|---:|---:|---:|---:|---:|---:|
| No sharing | 530.6 [525.2, 534.8] | 528.0 [524.7, 533.0] | 529.9 [525.3, 535.2] | 527.9 [524.9, 532.6] | 122.3 [121.4, 122.4] | 98.7 [98.2, 105.5] |
| Wide | 999.6 [997.8, 1001.2] | 312.5 [311.8, 313.8] | 321.3 [319.5, 325.1] | 313.7 [313.6, 314.6] | 217.5 [217.0, 218.1] | 286.5 [286.2, 287.4] |
| Unequal | 771.7 [771.2, 771.8] | 453.7 [450.1, 455.9] | 442.8 [442.7, 446.7] | 454.2 [451.9, 455.9] | 171.6 [169.8, 171.8] | 221.4 [219.1, 221.5] |
| Deep | 517.4 [516.2, 517.9] | 251.2 [249.6, 253.5] | 246.2 [243.6, 247.1] | 248.8 [248.7, 250.9] | 126.4 [126.1, 127.1] | 88.9 [87.6, 95.4] |
| 128 × 8192, shared | 13968.0 [13965.2, 13968.4] | 2526.3 [2514.7, 2526.4] | 2518.5 [2513.2, 2521.1] | 2522.6 [2513.7, 2523.0] | 2839.2 [2828.0, 2846.6] | 3879.8 [3875.5, 3880.9] |
| 128 × 8192, unshared | 38980.4 [38976.8, 38990.5] | 38981.3 [38975.2, 38991.7] | 38979.4 [38979.1, 39001.5] | 38984.0 [38968.0, 38991.1] | 2839.3 [2836.3, 2869.7] | 3844.5 [3838.1, 3895.3] |

The 128-request shared case is 1.126× faster than the faster vendor layout with dynamic grouping and 5.54× faster than per-request tree execution. Fixed and greedy grouping produce similar resident latency. The matched unshared case is 13.73× slower than the faster vendor operator. On the four smaller cases, at least one vendor layout is faster than every tree strategy.

| Shape | Fixed cold total, μs | Dynamic cold total, μs | Vendor BNSD cold total, μs | Vendor BSH cold total, μs |
|---|---:|---:|---:|---:|
| No sharing | 1757.7 | 2475.5 | 1328.8 | 1226.8 |
| Wide | 1055.7 | 4366.8 | 2478.3 | 2420.1 |
| Unequal | 1364.7 | 3294.9 | 2480.6 | 2076.3 |
| Deep | 823.0 | 4289.8 | 1298.9 | 1183.0 |
| 128 × 8192, shared | 7353.7 | 78733.5 | 39009.0 | 35548.3 |
| 128 × 8192, unshared | 80388.7 | 97527.7 | 39100.8 | 36582.9 |

Cold totals include layout conversion, CPU planning, device metadata, and attention. Both vendor layouts use a single batched index transfer and tensor gathers from the identical shared KV input. The long shared case favors fixed grouping when preparation is included: 7.35 ms versus 35.55 ms for the faster vendor cold path. Dynamic planning takes its total to 78.73 ms. Cold-input benefits depend on the plan choice and do not replace resident decode measurements. Raw records are in `results/benchmark_run{1,2,3}.json`; the plot is `results/attention_latency.pdf`.

## How tree shape affects reuse

Across all 81 shapes, dynamic resident attention has geometric-mean speedups of 1.315× over per-request execution, 1.008× over fixed grouping, and 1.005× over the local greedy method. The full plan-refresh path has corresponding ratios 0.797×, 0.377×, and 0.679×. Dynamic planning has no broad advantage over the simpler tree strategies.

The following table groups the complete grid by request count and shared-root length. Each entry is the geometric mean of per-request latency divided by dynamic resident latency over the nine remaining shape combinations.

| Requests | Root 32 tokens | Root 192 tokens | Root 768 tokens |
|---|---:|---:|---:|
| 4 | 1.026× | 1.099× | 1.325× |
| 12 | 1.119× | 1.281× | 1.569× |
| 24 | 1.303× | 1.449× | 1.855× |

Larger shared roots and more descendant queries increase useful reuse. The full range is 0.879×–2.641×. The slowest relative case has four requests, a 32-token root, and long private branches; the fastest has 24 requests, a 768-token root, and short private branches. Segment creation, partial outputs, and merging can outweigh limited KV reuse. In the 128-request long-context pair, fixed and dynamic sharing use 256 physical jobs; the unshared tree requires 2048. Each private query contributes four grouped-query rows, which are padded to a 16-row Cube tile. This padding and the larger partial-result set explain why the no-sharing path is poorly suited to this kernel.

The dynamic plan changes physical groups relative to fixed grouping on 58 of 81 trees. It improves the analytical cost over the per-child greedy plan on only one tree, by about 0.04%. That small choice difference is consistent with their nearly identical aggregate kernel measurements. Detailed results and the complete scatter plot are in `results/tree_grid.json` and `results/tree_grid.pdf`.

## Reusing decode plans

The 16-step experiment uses 32 requests, a 1024-token shared prefix, and 64 private prompt tokens per request. Both modes use the same reserved-tail physical layout and write the same growing contexts. Timings include actual CPU planning or device metadata updates, resident attention, and amortization of initial cached-plan construction.

| Strategy | Rebuild each step, μs | Cached plan, μs | Cached range, μs |
|---|---:|---:|---:|
| Fixed | 1400.1 | 536.4 | 527.2–607.6 |
| Greedy | 3125.9 | 563.8 | 562.9–569.2 |
| Dynamic | 3895.6 | 605.6 | 601.1–613.9 |

Plan reuse reduces the dynamic path's total by 6.43× on this workload. Fixed cached grouping remains the best of these choices. The experiment isolates the benefit of retaining immutable prompt groups from the choice of the initial grouping algorithm. Per-step records and the decomposition plot are in `results/decode_plan.json` and `results/decode_plan.pdf`.

## Public model workloads

The tokenizer and input records come from public Qwen3-4B and SQuAD 2.0 assets. Each workload has eight requests: shared-document questions (778–784 prompt tokens), dialogue histories (784–824), branch continuations (219–221), and distinct-prefix passages (72–397). Exact tokens are stored in `configs/qwen_workloads.json`.

`benchmark_qwen.py` measures complete fixed-length greedy generation after warmup, with rebuilding or cached plans. It records prefill, each decode step, total generation time, peak allocated memory, used and reserved KV positions, token agreement, and continuations.

## Pretrained Qwen3-4B generation

Six independent processes evaluate rebuilding and cached plans with seeds 401–403. Each eight-request batch produces 16 tokens per request: the first token follows prefill and the remaining 15 require decode steps. Both planning modes use identical reserved-tail physical layouts. Timings include plan construction or updates, all model layers, vocabulary selection, and synchronization. The prefill implementation is the original eager model path, applied separately to each request.

### Decode time

Milliseconds, median [minimum, maximum] across independent processes.

| Workload | Plan | Per request | Fixed | Greedy | Dynamic |
|---|---|---:|---:|---:|---:|
| document_qa | rebuild | 540.4 [536.0, 546.0] | 540.0 [524.2, 553.6] | 527.9 [520.9, 536.8] | 527.2 [524.1, 531.1] |
| document_qa | cached | 526.0 [524.5, 536.7] | 520.6 [516.4, 524.1] | 513.4 [508.7, 514.7] | 508.7 [504.4, 517.1] |
| multi_turn | rebuild | 538.3 [536.7, 552.9] | 525.7 [525.3, 543.6] | 540.4 [532.8, 543.3] | 545.4 [545.0, 550.1] |
| multi_turn | cached | 525.5 [519.9, 532.9] | 522.9 [518.6, 523.7] | 525.2 [524.0, 527.3] | 528.8 [523.7, 529.2] |
| multi_branch | rebuild | 490.2 [488.6, 495.2] | 508.4 [497.3, 521.1] | 508.9 [501.5, 509.6] | 513.2 [506.5, 515.8] |
| multi_branch | cached | 474.6 [474.2, 483.6] | 495.2 [490.6, 507.9] | 500.6 [489.2, 514.0] | 496.2 [491.0, 520.2] |
| no_share | rebuild | 495.0 [489.6, 495.3] | 490.8 [490.3, 498.9] | 506.4 [493.2, 518.6] | 494.8 [493.5, 502.2] |
| no_share | cached | 481.5 [480.3, 496.2] | 480.9 [479.9, 493.3] | 481.4 [480.5, 492.8] | 482.9 [479.6, 500.4] |

### Complete generation time

Milliseconds, median [minimum, maximum] across independent processes.

| Workload | Plan | Per request | Fixed | Greedy | Dynamic |
|---|---|---:|---:|---:|---:|
| document_qa | rebuild | 1213.4 [1206.1, 1223.0] | 1216.5 [1194.7, 1226.8] | 1202.6 [1191.3, 1213.1] | 1203.8 [1194.8, 1203.8] |
| document_qa | cached | 1192.0 [1191.1, 1216.5] | 1188.5 [1181.6, 1302.7] | 1179.4 [1175.7, 1192.6] | 1173.8 [1173.2, 1195.8] |
| multi_turn | rebuild | 1213.4 [1210.5, 1233.2] | 1205.8 [1202.4, 1217.2] | 1215.7 [1210.2, 1222.5] | 1220.2 [1219.8, 1229.9] |
| multi_turn | cached | 1194.3 [1189.6, 1214.5] | 1191.9 [1188.8, 1205.9] | 1194.5 [1193.5, 1209.1] | 1199.0 [1191.5, 1210.5] |
| multi_branch | rebuild | 846.9 [842.8, 847.9] | 863.2 [851.5, 877.9] | 862.0 [855.7, 866.6] | 869.3 [860.3, 870.7] |
| multi_branch | cached | 824.0 [822.3, 842.0] | 842.5 [840.6, 866.9] | 848.0 [838.8, 872.4] | 844.1 [842.0, 880.5] |
| no_share | rebuild | 839.0 [831.3, 839.4] | 834.8 [830.9, 842.9] | 850.3 [836.8, 860.2] | 838.4 [833.8, 845.7] |
| no_share | cached | 819.1 [818.0, 841.9] | 819.2 [816.6, 842.2] | 819.9 [817.2, 838.5] | 820.7 [817.8, 846.2] |

On shared-document questions, cached dynamic attention reduces complete decode time by 3.3% relative to cached per-request execution, and complete generation time by 1.5%. The multi-turn and unshared batches are close across strategies; grouped attention is slower on multi-branch generation. Reusing the dynamic plan improves its decode time by 2.4–3.5% across the four workloads. Full-model gains are smaller than isolated plan-update gains because projections, MLPs, and prefill occupy most of the runtime.

| Workload | Peak allocated GiB | Used physical KV positions | Logical KV positions | Token agreement with per-request generation |
|---|---:|---:|---:|---:|
| document_qa | 8.379 | 971 | 6366 | 100.00% |
| multi_turn | 8.413 | 1005 | 6516 | 100.00% |
| multi_branch | 7.784 | 353 | 1875 | 89.84% |
| no_share | 7.983 | 1393 | 1393 | 100.00% |

Peak allocated memory is equal across schedulers because all use the same shared pool allocation. The logical KV count includes the 15 generated tokens whose KV values have been computed; the sixteenth output token is selected but has not been fed back. Exact token IDs and continuations are retained for every run. All three processes and both planning modes produce the same agreement values. Raw measurements are in `results/qwen4b_{rebuild,cached}_run{1,2,3}.json`; the complete aggregate and plot are `results/qwen4b.json` and `results/qwen4b.pdf`.

### Common-history generation divergence

The multi-branch workload first diverges at generated-token index 9 for requests 2 and 3. Replaying the identical preceding token history yields the following top-two logits:

| Request | Per-request choices and logits | Fixed-group choices and logits | Full-sequence choices and logits |
|---|---|---|---|
| 2 | “made” 19.625, “engaged” 19.625 | “engaged” 19.750, “made” 19.625 | “engaged” 19.625, “made” 19.500 |
| 3 | “conducted” 19.875, “attacked” 19.875 | “attacked” 20.000, “conducted” 19.750 | “attacked” 20.000, “conducted” 19.750 |

Both per-request choices are exact BF16 ties. On this same history, all 72 layer comparisons against independent FP32 attention pass: maximum relative L2 errors are 0.191% for per-request attention and 0.218% for fixed grouping, versus 0.221% for vendor attention on the same Q/K/V. The numerical paths resolve near-tied logits differently, and autoregressive feedback amplifies the token difference. The full-sequence BF16 path agrees with fixed grouping for these two requests; across all eight requests, full-sequence top-token agreement is 62.5% for the per-request path and 87.5% for fixed grouping. Exact histories, layer errors, and top-two values are in `results/qwen4b_divergence.json`.

## Limitations

The cost model's roofline constants are analytical estimates rather than fitted latency predictions. The dynamic program minimizes that model within its tree-partition choices. Kernel measurements show no general advantage over fixed or greedy tree grouping, and the vendor resident operator is faster on five of the six representative shapes. The model adapter supports a fixed batch, full attention, and single-token decode on one 910B; prefill retains the original model path. The pretrained generation workload contains four eight-request batches and 16 generated tokens per request; it establishes integration and timing rather than broad model-quality evaluation. BF16 execution paths can produce different greedy continuations at near-tied logits; the measured multi-branch token agreement is 89.84%. Synthetic peak memory includes resident reference layouts and is not a production memory-savings estimate. All reported hardware conclusions are single-device conclusions.

## Sources and reproduction

The reference artifact is [FastTree](https://github.com/PanZaifeng/FastTree-Artifact), pinned at `167928b56513d8ac4ae4948da70b9f6ef07d8f5f`. The Qwen3-4B weights come from the [official ModelScope repository](https://modelscope.cn/models/Qwen/Qwen3-4B); public file revisions are retained in `configs/model_source.json`. The model and workload references are [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/). Environment versions are in `configs/environment.json`; commands are in `README.md`.
