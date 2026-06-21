# CUDA Kernel-Writing RL Env Tasks

This branch contains two HUD RL environment tasks for testing an agent's ability to write and optimize custom CUDA kernels against golden references.

## Tasks

### `cuda_fi_block`

A TorchTitan DeepSeek-style inference block task. The agent starts from a correct but slow PyTorch fallback and must implement raw CUDA fused ops for the candidate path. Graders check CUDA extension usage, FlashInfer avoidance, parity against eager plus a tuned V2 golden reference, and speed against that golden reference.

More detail: [`README_cuda_fi_ffn_V2_impl.md`](README_cuda_fi_ffn_V2_impl.md)

### `sglang_dsv4_indexer`

An SGLang DeepSeek V4 indexer micro-kernel task. The agent implements RoPE, Hadamard transform, FP8 quantization, and weight-scale output in CUDA. Graders compare against the staged SGLang JIT path and a hidden golden CUDA reference.

More detail: [`README_dsv4_indexer_task.md`](README_dsv4_indexer_task.md)

## Rollouts

Agent rollout logs and grader summaries are under [`rollouts/`](rollouts/). They document both successful and partial solutions.

Examples:

- `cuda_fi_block`: Opus 4.8 solves the fixed task cleanly; Kimi openai-compatible runs expose the missing write/shell-tool issue on that agent path.
- `sglang_dsv4_indexer`: GPT-5.5 run 6 reaches full reward after installing the SGLang JIT dependency (`tvm_ffi`). Opus 4.5 produces a valid CUDA solution but does not receive full hidden-golden reward because it misses the stronger golden topology: more element-wise parallelism, 128 threads per row, shared-memory coordination, and less redundant output writing.

## Hackathon note

These two tasks test CUDA-writing capabilities: fusion, optimization, parity against reference implementations, and speed against hidden golden kernels.

The intended longer-term direction is to automatically generate golden reference kernel sets through a framework such as TileRT, CODA kernels, or similar, expose profiler traces to the agent, and train agents to reach near-optimal kernels within limited iterations through hardware-aware reasoning. The asymmetry is intentional: the framework can use well-tuned generated or hand-crafted kernels, while the agent must compress that design space into a few edits and benchmark iterations.

I tested locally on a laptop 5070 Ti after hitting cloud GPU setup issues. I also tried fine-tuning Kimi K2.7 from local rollouts, but HUD's `openai_compatible` agent path did not expose file-writing/shell tools, so that RL signal could not be verified for Kimi through that route. The limited tool catalog is visible in [`hud/agents/openai_compatible/tools/__init__.py`](https://github.com/hud-evals/hud-python/blob/main/hud/agents/openai_compatible/tools/__init__.py). Since KernelBench-style tasks are not saturated, these environments should still have learnable signal once the model/tool interface is fixed.
