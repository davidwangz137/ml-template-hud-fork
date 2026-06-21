# `cuda_fi_block` V2 implementation notes

## Purpose

`cuda_fi_block` is the V2 step after `cuda_fi_ffn`.

V1 (`cuda_fi_ffn`) proves an agent can write and load raw CUDA kernels for two isolated fused ops:

```text
fused residual-add RMSNorm
SwiGLU silu(gate) * up
```

V2 (`cuda_fi_block`) makes the task more realistic: one inference-only DeepSeek-style decoder block with attention, RoPE, residuals, FFN, a FlashInfer reference path, and a raw-CUDA candidate path.

It is still not a full TileRT-class whole-model inference task. It is a bridge task: complex enough to require profiling and launch/layout decisions, but constrained enough that a handwritten CUDA reference solution can pass.

## Files

Task scaffold:

```text
tasks/cuda_fi_block/
  task.py
  __init__.py
  00_cuda_flashinfer_deepseek_block.patch
  reference_solution.diff
  reference_solution_files/cuda_fusion_block/
```

Reference/source-of-truth implementation:

```text
tasks/cuda_fi_block/reference_solution_files/cuda_fusion_block/
  __init__.py
  deepseek_block.py
  benchmark_deepseek_block.py
  custom_ops.py
  cuda_ext/
    README.md
    block_kernels.cpp
    block_kernels.cu
```

Graders:

```text
tasks/graders/check_cuda_block_extension.py
tasks/graders/check_cuda_block_no_flashinfer.py
tasks/graders/check_cuda_block_parity.py
tasks/graders/check_cuda_block_speed.py
```

## Running locally

Use a clean clone if you want to preserve this working tree:

```bash
git clone https://github.com/davidwangz137/ml-template-hud-fork.git ml-template-hud-fork-v2
cd ml-template-hud-fork-v2
git checkout v2_cuda_task
```

Run the HUD task locally through the checked-in local runner:

```bash
uv run python local_runner.py \
  --task cuda_fi_block \
  --model gpt-5.5 \
  --max-steps 30 \
  --timeout 2400 \
  --stream-steps /tmp/local_run_cuda_fi_block_steps.jsonl
```

Recommended monitor command in another terminal:

```bash
watch -n 120 'printf "== processes ==\n"; \
pgrep -af "local_runner.py --task|hud.environment.server .*env.py|python .*benchmark_deepseek_block|nvcc|c\\+\\+|ptxas" || true; \
printf "\n== gpu ==\n"; nvidia-smi; \
printf "\n== recent steps ==\n"; tail -20 /tmp/local_run_cuda_fi_block_steps.jsonl 2>/dev/null; \
printf "\n== workspace files ==\n"; ls -lh /tmp/hud_workspace/torchtitan/experiments/cuda_fusion_block 2>/dev/null'
```

Expected healthy behavior:

- workspace stages under `/tmp/hud_workspace`;
- the agent edits only `torchtitan/experiments/cuda_fusion_block/**`;
- CUDA extension build invokes `nvcc` / `ptxas`;
- benchmark runs `benchmark_deepseek_block.py`;
- final graders run:
  - `check_cuda_block_extension`;
  - `check_cuda_block_no_flashinfer`;
  - `check_cuda_block_parity`;
  - `check_cuda_block_speed`;
- reward may be partial because speed is continuous.

The local runner uses the repo `.venv` through `HUD_LOCAL_VENV` and binds local NVIDIA devices into the HUD workspace. It is not Docker or Modal.

## HUD staging model

The actual task patch is:

```text
tasks/cuda_fi_block/00_cuda_flashinfer_deepseek_block.patch
```

It stages this package into the agent workspace:

```text
torchtitan/experiments/cuda_fusion_block/
```

The staged CUDA files are stubs. The candidate must replace them.

The known-good reference solution is separate:

```text
tasks/cuda_fi_block/reference_solution.diff
```

That diff applies on top of the staged task patch and replaces only the CUDA extension stubs with real kernels. It is intentionally not named `*.patch`, because HUD task setup loads every `*.patch` in the task directory.

## Block computation

The block is a dense, inference-only DeepSeek/Qwen-style decoder block:

```text
x
 -> attention RMSNorm
 -> QKV projection
 -> RoPE on Q/K
 -> causal GQA attention
 -> output projection
 -> residual add
 -> FFN RMSNorm
 -> gate/up projection
 -> SwiGLU
 -> down projection
 -> residual add
```

In code, the three paths are in:

```text
deepseek_block.py
```

They expose:

```python
eager_forward(x, weights, cos, sin)
compiler_forward(x, weights, cos, sin)
candidate_forward(x, weights, cos, sin)
```

## Reference paths

### `eager_forward`

Readable PyTorch correctness oracle.

Uses:

- PyTorch RMSNorm;
- separate Q/K/V GEMMs;
- Python/PyTorch RoPE;
- PyTorch SDPA;
- separate gate/up GEMMs;
- PyTorch SwiGLU.

This is the semantic baseline.

### `compiler_forward`

FlashInfer-backed reference path.

Uses:

- packed QKV weight;
- FlashInfer single prefill attention;
- FlashInfer fused residual-add RMSNorm;
- packed gate/up weight;
- FlashInfer SwiGLU;
- warm `torch.compile(..., mode="reduce-overhead")` wrapping the FlashInfer/PyTorch reference block when possible.

This is called `compiler_forward` because it stands in for the high-performance library/compiler path. It is forbidden from the candidate path.

Fairness rule: if the candidate uses warm `torch.compile`, the FlashInfer reference gets the same warm compile wrapper. In practice, FlashInfer's Python/JIT/custom-op stack causes graph breaks and warnings, so `torch.compile` helps it less than it helps the candidate. That is still the right comparison: both paths get the same inference graph-capture opportunity, and timing excludes warmup/compile.

### `candidate_forward`

Agent-owned path.

The reference solution uses:

- raw CUDA attention RMSNorm;
- packed QKV weight;
- raw CUDA RoPE;
- PyTorch SDPA;
- PyTorch output projection;
- PyTorch residual add + FFN RMSNorm for the fastest profiled variant;
- packed gate/up weight;
- raw CUDA SwiGLU;
- PyTorch down projection;
- warm `torch.compile(..., mode="reduce-overhead")` wrapping the candidate block.

GEMMs and SDPA remain PyTorch in V2. This is deliberate: the task targets fusion/layout around GEMMs and attention, plus inference graph capture/launch reduction, not reimplementing cuBLAS or FlashAttention from scratch.

## Exact optimization list in the current V2 reference

This is the current source-of-truth strategy in `reference_solution_files/cuda_fusion_block/deepseek_block.py`.

### Kept as library ops

- GEMMs remain PyTorch matmul:
  - `normed @ w_qkv`;
  - `attn @ w_o`;
  - `normed @ w_gate_up`;
  - `hidden @ w_down`.
- Attention remains PyTorch SDPA in the candidate path.
- FFN residual add + second RMSNorm uses PyTorch in the fastest profiled variant, because it beat the custom fused version once the whole block was compiled.

### Custom CUDA ops

- Attention input RMSNorm:
  - `custom_ops.rmsnorm`;
  - one CTA per token row;
  - FP32 reduction, BF16 output.
- RoPE:
  - `custom_ops.apply_rope_`;
  - in-place Q/K rotation;
  - separate internal launches for Q and K.
- SwiGLU:
  - `custom_ops.silu_and_mul`;
  - operates on packed `[gate, up]`;
  - FP32 sigmoid/Silu math, BF16 output.

The reference extension also implements `fused_add_rmsnorm_`, but the fastest profiled V2 candidate path currently does not use it for the FFN norm. It remains part of the task API because it is useful, tests raw CUDA ability, and may win on other shapes or under different compile behavior.

### Layout choices

- Candidate uses packed QKV:
  ```python
  q, k, v = split(rmsnorm(x) @ w_qkv)
  ```
- Candidate uses packed gate/up:
  ```python
  gate_up = ffn_norm(residual) @ w_gate_up
  hidden = silu_and_mul(gate_up)
  ```
- Packing reduces GEMM launch count and gives `torch.compile` a larger graph to capture.

### Graph/launch optimization

- Candidate is wrapped in:
  ```python
  torch.compile(_candidate_impl, mode="reduce-overhead", fullgraph=False)
  ```
- Benchmark warmup runs happen before CUDA-event timing.
- `torch.compiler.cudagraph_mark_step_begin()` is called before compiled invocation.
- The compiled output is cloned before returning to avoid CUDAGraph output overwrite hazards in parity/timing loops.

### Profiled choices that lost locally

These were tested and rejected for the V2 reference:

- packed QKV without compile;
- packed gate/up without compile;
- custom fused residual-add RMSNorm for the FFN norm inside the compiled block;
- all-custom non-GEMM path without `torch.compile`;
- separate Q/K/V plus separate gate/up after compile.

The fastest local profile was: custom attention RMSNorm + packed QKV + custom RoPE + PyTorch SDPA + PyTorch FFN residual/RMSNorm + packed gate/up + custom SwiGLU + warm `torch.compile`.

## Why packed QKV and gate/up matter

The first reference version used separate GEMMs:

```python
q = normed @ w_q
k = normed @ w_k
v = normed @ w_v
gate = normed @ w_gate
up = normed @ w_up
```

That caused extra launch overhead and made the candidate slower than necessary.

The improved V2 reference packs these:

```python
w_qkv = cat(w_q, w_k, w_v)
w_gate_up = cat(w_gate, w_up)
```

Then the optimized paths use:

```python
q, k, v = split(normed @ w_qkv)
gate_up = normed @ w_gate_up
```

This is the kind of layout decision the task is meant to reward. It is not a custom CUDA kernel by itself, but it changes the surrounding kernel/launch profile and makes the raw CUDA fused ops more worthwhile.

## Raw CUDA extension API

Candidate-owned public API in `custom_ops.py`:

```python
rmsnorm(input, weight, eps) -> Tensor
fused_add_rmsnorm_(input, residual, weight, eps) -> None
apply_rope_(q, k, cos, sin) -> None
silu_and_mul(gate_up, out=None) -> Tensor
```

The C++ bindings are in:

```text
cuda_ext/block_kernels.cpp
```

The CUDA kernels are in:

```text
cuda_ext/block_kernels.cu
```

## CUDA kernels in the reference solution

### `rmsnorm_kernel`

Input:

```text
input:  [rows, dim] bf16
weight: [dim] bf16
out:    [rows, dim] bf16
```

Mechanics:

- one CTA per row;
- 256 threads;
- FP32 sum of squares;
- shared-memory reduction;
- BF16 output.

This handles attention RMSNorm.

### `fused_add_rmsnorm_kernel`

Input:

```text
input:    [rows, dim] bf16  # attention output, overwritten with normalized residual
residual: [rows, dim] bf16  # x, updated in place
weight:   [dim] bf16
```

Semantics:

```text
residual += input
input = rmsnorm(residual) * weight
```

This matches FlashInfer `fused_add_rmsnorm` semantics.

### `rope_kernel`

Input:

```text
q:   [1, seq, q_heads, head_dim] bf16
k:   [1, seq, kv_heads, head_dim] bf16
cos: [seq, head_dim / 2] fp32
sin: [seq, head_dim / 2] fp32
```

Mechanics:

- grid-stride over `(seq, heads, head_dim / 2)`;
- launched once for Q and once for K;
- in-place rotation.

### `silu_and_mul_kernel`

Input:

```text
gate_up: [rows, 2 * hidden_dim] bf16
out:     [rows, hidden_dim] bf16
```

Semantics:

```text
out = silu(gate_up[:, :hidden_dim]) * gate_up[:, hidden_dim:]
```

Uses FP32 sigmoid/Silu math and casts to BF16.

## Grading mechanics

### `check_cuda_block_extension.py`

Checks:

- real `.cu` and C++ binding files exist;
- stubs were replaced;
- CUDA source contains `__global__`;
- binding source contains `PYBIND11_MODULE`;
- candidate-owned code does not reference FlashInfer;
- candidate imports and runs a smoke shape.

### `check_cuda_block_no_flashinfer.py`

Checks:

- static scan for FlashInfer references in candidate-owned files;
- runtime monkeypatch blocks `flashinfer`;
- `candidate_forward` still runs.

### `check_cuda_block_parity.py`

Hidden shapes compare candidate against both:

- `eager_forward`;
- `compiler_forward`.

Current tolerances:

```text
max_abs <= 0.12
mean_abs <= 0.02
```

These are BF16-friendly and allow small differences between FlashInfer/PyTorch/custom CUDA accumulation order.

### `check_cuda_block_speed.py`

Times full block paths with CUDA events.

Current pass thresholds:

```text
candidate <= 0.90 * eager
candidate <= 1.30 * compiler
```

Timing excludes compile/warmup. Both `compiler_forward` and `candidate_forward` attempt `torch.compile(..., mode="reduce-overhead")`; warmup iterations absorb graph capture/compilation, and the timed loop measures steady-state execution. The candidate still has to pass with FlashInfer blocked, so it cannot win by calling the reference library.

## Clean validation result

Reference solution passes all block graders:

```text
block extension passed
block candidate runs with FlashInfer blocked
block parity passed
block speed passed
```

Representative clean speed-grader timing after fair compile integration:

```text
seq=257 dim=768 heads=12 kv_heads=3 hidden=2048
  eager_ms=0.6347
  compiler_ms=0.5300
  candidate_ms=0.3231
  candidate_vs_eager=1.964x
  candidate_vs_compiler=1.640x

seq=384 dim=768 heads=12 kv_heads=3 hidden=2048
  eager_ms=1.0645
  compiler_ms=0.8702
  candidate_ms=0.3201
  candidate_vs_eager=3.325x
  candidate_vs_compiler=2.718x
```

This is now a much stronger optimized reference for V2. It is still not a theoretical optimum: attention uses PyTorch SDPA rather than a custom FlashAttention-style kernel, GEMMs are generic PyTorch GEMMs, and there is no GEMM epilogue fusion. But for these V2 shapes, warm compile applied to both sides plus custom CUDA fused ops and layout choices beats eager and the FlashInfer reference path.

## Adopted faster HUD-agent reference

After the first local HUD run, the gpt-5.5 agent produced a faster legal candidate than the previous hand-written reference. That solution is now copied into:

```text
tasks/cuda_fi_block/reference_solution_files/cuda_fusion_block/
```

Backups of the previous reference are kept beside it:

```text
deepseek_block_old.py
cuda_ext/block_kernels_old.cpp
cuda_ext/block_kernels_old.cu
```

Key additional optimization:

- explicit candidate-side CUDA graph replay instead of relying only on `torch.compile`;
- graph cache keyed by static activation shape, dtype, weight pointers, RoPE pointers, and epsilon;
- two captured graph slots to avoid clobbering the immediately previous returned tensor;
- static input copy + graph replay in steady state.

This is a valid inference optimization under the task rules: it does not call FlashInfer, it preserves parity, and warmup absorbs graph capture before timing.

Current adopted-reference speed-grader run:

```text
seq=257 dim=768 heads=12 kv_heads=3 hidden=2048
  eager_ms=0.6320 compiler_ms=0.6957 candidate_ms=0.1587
  candidate_vs_eager=3.983x candidate_vs_compiler=4.384x

seq=384 dim=768 heads=12 kv_heads=3 hidden=2048
  eager_ms=0.7526 compiler_ms=0.8031 candidate_ms=0.2273
  candidate_vs_eager=3.311x candidate_vs_compiler=3.534x

geomean_speedup_vs_eager=3.6315
geomean_vs_compiler=3.9361
```

## Continuous speed reward

`check_cuda_block_speed.py` now emits a continuous partial score:

```text
SCORE: <0..1>
```

The environment supports `score_stdout=True` graders and parses that line into the subscore value.

Current formula:

```text
geomean_speedup = geometric_mean(eager_ms / candidate_ms over hidden shapes)
speed_score = clamp(geomean_speedup / 5.0, 0.0, 1.0)
```

Reward-hacking guard:

```text
if any shape reports >50x speedup over eager:
  fail the speed grader
```

Correctness/provenance are still separate graders:

- real CUDA extension;
- no FlashInfer dependency in candidate;
- numeric parity vs eager and compiler references.

This means a future agent can get more reward than the current adopted reference if it produces a faster legal implementation, up to the cap.

## Why V2 is more complex than V1

V1 has two obvious kernels. V2 has interacting choices:

- separate vs packed QKV GEMMs;
- separate vs packed gate/up GEMMs;
- PyTorch SDPA vs FlashInfer attention;
- which RMSNorms to fuse;
- whether RoPE should be custom CUDA or PyTorch;
- whether launch count or memory bandwidth dominates;
- hidden shape sensitivity.

An agent should profile and iterate rather than blindly write kernels.

## Known limitations

V2 still does not include:

- custom attention kernel / FlashAttention-style online softmax;
- custom GEMM kernels;
- GEMM epilogue fusion;
- KV-cache decode loop;
- paged KV cache;
- MoE routing;
- expert paging;
- full model logits;
- trained/router-calibrated behavior.

It is one block, one request, prefill-style attention, random weights.

## What a stronger-than-reference V2 agent could still implement

The current reference is intentionally aggressive but still leaves several “crazy intelligence” upgrades on the table. These are valid directions for an agent that has strong CUDA/kernel recall and enough time.

### Custom FlashAttention-style prefill

Replace PyTorch SDPA with a fixed-shape causal GQA attention kernel.

Core mechanics:

```text
for each query block and head:
  stream over K/V tiles
  apply causal mask
  maintain online softmax max/sum
  accumulate weighted V
  write output
```

Important details:

- support GQA: multiple Q heads share each KV head;
- specialize `head_dim=64`;
- use FP32 accumulators for softmax statistics;
- keep Q/K/V in BF16;
- avoid materializing attention scores;
- tile over sequence to keep K/V in shared memory where useful;
- handle non-power-of-two `seq` like 257.

This is the biggest missing V2 optimization. It would remove the largest remaining dependency on PyTorch SDPA and make the task much closer to FlashInfer/TileRT territory.

### Fused RoPE + attention input path

Current candidate does:

```text
QKV GEMM -> separate RoPE kernel -> SDPA
```

A stronger implementation could fold RoPE into the attention kernel load path:

```text
load Q/K tile
apply RoPE in registers
compute attention tile
```

That removes the standalone RoPE launch and Q/K global-memory rewrite.

### GEMM epilogue fusion

Current V2 uses generic PyTorch GEMMs, so it cannot fuse work into GEMM epilogues.

Potential epilogues:

- QKV projection epilogue writes Q/K/V directly into attention layout;
- output projection epilogue adds residual or prepares RMSNorm statistics;
- gate/up projection epilogue applies SwiGLU before global write;
- down projection epilogue adds final residual.

This likely requires CUTLASS/CuTe/Triton-level work or hand-written limited-shape matmul kernels. It is beyond the reference, but it is exactly the CODA/TileRT-style direction.

### Persistent/static decode graph

The reference uses `torch.compile(..., mode=\"reduce-overhead\")`, which is a generic graph-capture win.

A stronger implementation could manually create a stable inference engine:

```text
allocate all intermediates once
capture CUDA graph after warmup
replay graph for timing
```

This avoids allocator and Python dispatch overhead more explicitly than relying on `torch.compile`.

### Buffer reuse and no-clone API

The benchmark clones the input activation per timed iteration to keep paths independent. A production inference engine would maintain reusable buffers and mutate/overwrite intermediates in a fixed schedule.

Potential improvement:

- expose `CandidateBlockEngine`;
- preallocate QKV, attention output, gate/up, hidden buffers;
- run `forward_into(x, out)` without fresh allocations;
- benchmark graph replay over stable buffers.

This would make V2 closer to real inference, but changes the API enough that it should be a separate hardening step.

### More precise custom norm kernels

The current norm kernels are simple one-CTA-per-row reductions. A stronger version could:

- vectorize BF16 loads/stores;
- use warp-level reductions before shared memory;
- specialize common dims like 768/1024/1536;
- fuse scale/writeback more aggressively;
- reduce conversion overhead.

This is smaller than custom attention but still useful.

### Better task-hardening stance

If we want V2 to reward these harder upgrades, the speed grader can eventually tighten to something like:

```text
candidate <= 0.70 * eager
candidate <= 1.05 * compiler
```

Only do this after a stronger reference solution with custom attention or graph replay proves those thresholds are reachable.

## How V3 should change

V3 should introduce the mechanics that make profiling unavoidable.

Best next task: `cuda_fi_moe_trace`.

### V3 target computation

```text
x:              [tokens, dim]
expert_ids:     [tokens, top_k]
expert_weights: [tokens, top_k]
expert params:  [num_experts, ...]

count tokens per expert
compute expert offsets
pack selected token states by expert
run expert gate/up/SwiGLU/down
weight expert outputs
scatter/combine back to token order
```

### Why V3 is meatier

It adds multiple competing bottlenecks:

- routing/counting;
- prefix sums / offsets;
- token pack/reorder;
- per-expert small GEMMs;
- SwiGLU fusion;
- combine/scatter;
- top-k weighting;
- launch count;
- memory traffic;
- hot/cold expert distribution.

A naive correct implementation can pass parity but fail speed. A good implementation must profile.

### V3 visible benchmark output

The benchmark should print component timings:

```json
{
  "route_count_ms": 0.08,
  "prefix_offsets_ms": 0.03,
  "pack_tokens_ms": 0.22,
  "expert_compute_ms": 1.41,
  "combine_scatter_ms": 0.31,
  "total_ms": 2.05
}
```

The prompt should explicitly say:

```text
Run the benchmark first.
Find the largest component gap.
Optimize one kernel/layout.
Rerun.
Repeat until hidden speed thresholds pass.
```

### V3 route traces

Do not use random router logits for expert-paging tasks. Use deterministic traces:

1. uniform trace for correctness;
2. skewed trace for hot experts;
3. temporal-locality trace for cache behavior.

Example:

```text
num_experts=16
top_k=2
tokens=2048
experts 0..3 receive 70% of assignments
neighboring decode steps keep 60% of expert choices stable
```

### V3 reference path

Need both:

- eager PyTorch oracle;
- high-performance reference path, ideally FlashInfer grouped GEMM or TileRT-supported MoE op where available;
- handwritten CUDA reference solution proving thresholds are reachable.

If TileRT is used, keep it constrained to supported op templates. Public TileRT is not an arbitrary graph compiler.

### V3 likely custom kernels

Candidate/reference solution should likely implement:

- histogram/count experts;
- prefix offset computation or CPU-precomputed visible baseline with hidden GPU requirement;
- pack tokens by expert;
- fuse top-k weight multiply with scatter/combine;
- SwiGLU elementwise fusion;
- skip empty experts;
- maybe specialize `top_k=2`.

V3 can still allow PyTorch GEMMs initially. Later variants can require grouped GEMM or per-expert custom matmul for small expert dims.

## Practical hardening strategy

Use V2 to calibrate agent ability.

Then V3 can be hardened by limiting:

- wall time;
- number of tool calls/steps;
- visible benchmark shapes;
- allowed dependencies;
- amount of profile hint detail.

The known-good V2 reference took explicit profiling/layout iteration. A less capable agent should get correctness but miss speed; a strong agent should discover packed projections and fused kernels quickly.
