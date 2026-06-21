# DSV4 Indexer HUD Task

This documents the SGLang DeepSeek-V4 indexer HUD prototype added in this repo.

## What this task implements

The task extracts a real SGLang DeepSeek-V4 serving subgraph from `sglang.jit_kernel.dsv4`:

```text
q_input[B, H, 128]
  -> RoPE on final 64 dims
  -> SGLang lane-major 128-point Hadamard
  -> FP8 E4M3 per-row quantization
  -> weights_out = weight * weight_scale * activation_scale
```

The task compares four implementations:

- `eager_forward`: PyTorch semantic reference.
- `compiler_forward`: `torch.compile(eager_forward)` baseline.
- `sglang_forward`: SGLang DSV4 JIT kernel oracle/baseline.
- `candidate_forward`: candidate-owned CUDA extension.

SGLang is the correctness oracle because PyTorch FP8 conversion does not exactly match SGLang/CUDA FP8 packing. In observed runs, `eager` differs from SGLang by `q_max=32.0`, while a correct CUDA candidate matches SGLang at `q_max<=0.00390625`.

## Files

Task files:

```text
tasks/sglang_dsv4_indexer/__init__.py
tasks/sglang_dsv4_indexer/task.py
tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer/__init__.py
tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer/indexer.py
tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer/custom_ops.py
tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer/cuda_ext/indexer_kernels.cpp
tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer/cuda_ext/indexer_kernels.cu
tasks/sglang_dsv4_indexer/reference_solution_files/dsv4_indexer/__init__.py
tasks/sglang_dsv4_indexer/reference_solution_files/dsv4_indexer/indexer.py
tasks/sglang_dsv4_indexer/reference_solution_files/dsv4_indexer/custom_ops.py
tasks/sglang_dsv4_indexer/reference_solution_files/dsv4_indexer/cuda_ext/indexer_kernels.cpp
tasks/sglang_dsv4_indexer/reference_solution_files/dsv4_indexer/cuda_ext/indexer_kernels.cu
```

Graders:

```text
tasks/graders/check_dsv4_indexer_speed.py
tasks/graders/check_dsv4_indexer_golden_speed.py
```

Standalone development benchmark:

```text
scripts/benchmark_sglang_dsv4_indexer_subgraph.py
```

Rollout artifacts worth copying:

```text
rollouts/local_run_sglang_dsv4_steps4.jsonl
rollouts/local_run_sglang_dsv4_trace4.json
rollouts/local_run_sglang_dsv4_grader4.txt
rollouts/local_run_sglang_dsv4_summary4.txt
rollouts/local_run_sglang_dsv4_golden_compare4.txt
```

Run 3 artifacts are useful only as a failed-run diagnosis showing the earlier eager-fallback bug:

```text
rollouts/local_run_sglang_dsv4_steps3.jsonl
rollouts/local_run_sglang_dsv4_trace3.json
rollouts/local_run_sglang_dsv4_grader3.txt
rollouts/local_run_sglang_dsv4_summary3.txt
```

Do not copy `__pycache__` files.

## External dependency

The task setup stages the local SGLang JIT kernel source into the HUD workspace:

```text
../sglang/python/sglang/jit_kernel -> $WORKSPACE/sglang_jit_kernel
```

So the target checkout should also have SGLang cloned at:

```text
~/code_hackathon/sglang
```

or `tasks/sglang_dsv4_indexer/task.py` should be adjusted to point at the local SGLang checkout.

## Running the grader directly

From repo root:

```bash
.venv/bin/python tasks/graders/check_dsv4_indexer_speed.py .
```

This scores the repository reference implementation under `tasks/sglang_dsv4_indexer/reference_solution_files`.

Expected strong reference behavior on the current RTX 5070 Ti Laptop GPU:

```text
candidate_vs_sglang_q_max <= 0.00390625
candidate_vs_sglang_w_max <= 1.2e-10
geomean_vs_sglang roughly 8x
SCORE: 1.000000
```

To test a HUD candidate workspace:

```bash
.venv/bin/python tasks/graders/check_dsv4_indexer_speed.py /tmp/hud_workspace_sglang_dsv4_4
```

To compare a candidate workspace against the hidden golden/reference CUDA implementation:

```bash
.venv/bin/python tasks/graders/check_dsv4_indexer_golden_speed.py /tmp/hud_workspace_sglang_dsv4_4 .
```

The hidden-golden grader runs the public speed grader twice with isolated `TORCH_EXTENSIONS_DIR` values:

1. candidate workspace;
2. `tasks/sglang_dsv4_indexer/reference_solution_files`.

It prints:

```text
candidate_geomean_vs_sglang=...
golden_geomean_vs_sglang=...
candidate_vs_hidden_golden=...
SCORE: ...
```

## Running through HUD local runner

Example:

```bash
uv run python local_runner.py \
  --task sglang_dsv4_indexer \
  --model gpt-5.5 \
  --max-steps 30 \
  --timeout 2400 \
  --workspace /tmp/hud_workspace_sglang_dsv4_4 \
  --stream-steps /tmp/local_run_sglang_dsv4_steps4.jsonl
```

Current task scoring:

```text
70%  check_dsv4_indexer_speed
30%  check_dsv4_indexer_golden_speed
```

This prevents agents from receiving full credit for merely beating SGLang if they remain well below the hidden CUDA golden.

## Known harness fixes

Two fixes were required before the task became meaningful:

1. `custom_ops.py` now prepends `sys.prefix/bin` to `PATH` before extension build so `ninja` is discoverable inside HUD.
2. CUDA extension build failures now raise when CUDA is available. The initial harness silently fell back to eager, which let invalid/slow solutions pass loose parity.

The public parity threshold was tightened to:

```python
candidate_vs_sglang_q_max <= 0.01
candidate_vs_sglang_w_max <= 1e-6
```

## Run 4 result

Run 4 (`gpt-5.5`) produced a valid optimized CUDA solution:

```text
Reward: 1.0
Stop reason: done

shape 64x16:
  sglang_ms=0.1147
  candidate_ms=0.0146
  cand_vs_sglang=7.879

shape 256x16:
  sglang_ms=0.1125
  candidate_ms=0.0149
  cand_vs_sglang=7.547

geomean_vs_sglang=7.7110
SCORE=1.000000
```

Posthoc hidden-golden comparison showed the hand-written reference remained faster overall in one rerun:

```text
run-4 candidate_geomean_vs_sglang=5.3330
hidden golden_geomean_vs_sglang=8.9999
candidate_vs_hidden_golden=0.5926
```

That motivated adding `check_dsv4_indexer_golden_speed.py` as a second grader.

## Planned copy target

When ready, copy these files into:

```text
~/code_hackathon/ml-template-hud-fork2
```

Preserve relative paths. Do not copy generated `__pycache__` files.
