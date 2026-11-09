# Research project

**Tree-shared attention and reusable decode schedules on Ascend 910B**

- Built compressed-prefix scheduling, a per-child greedy baseline, and subset dynamic programming with Cube alignment and physical KV continuity.
- Developed Triton Ascend matrix attention and stable global-softmax merging; integrated public pretrained Qwen3-4B with shared KV pools and reusable prompt plans, validated by 1,224 layer comparisons against independent FP32 attention.
- Evaluated all 81 held-out tree shapes in three independent processes, achieving 1.315× geometric-mean resident speedup over per-request execution. On 128 requests with 8192-token contexts sharing 8128 tokens, the kernel took 2.523 ms versus 2.839 ms for the faster vendor layout.
- Reduced planning-plus-attention time from 3.896 ms to 0.606 ms per step through plan reuse in a 32-request, 16-step experiment; fixed cached grouping reached 0.536 ms.
- Completed three-process Qwen3-4B generation measurements on four public-text workloads. Cached tree attention reduced document-question decode time by 3.3%; same-history layer checks and top-two logits explained observed BF16 continuation differences.

The study quantifies how tree shape and scheduling overhead determine useful KV reuse. Fixed grouping matches the dynamic program's aggregate resident performance with less planning work. Long shared contexts benefit, while unshared contexts favor the vendor fused operator. The implementation and measurements separate shared computation, initial planning, incremental updates, layout conversion, and complete model execution.
