from env import SRC_DIR as S, WORKSPACE as W, restore_reference_parity
from tasks.graders import grader


_PROMPT = """Optimize a DeepSeek-V4/SGLang indexer subgraph with a custom CUDA implementation.

Target files:
  dsv4_indexer/custom_ops.py
  dsv4_indexer/cuda_ext/*

The subgraph is:
  q_input[B,H,128] -> RoPE on final 64 dims -> 128-point Hadamard -> FP8 E4M3 quant -> weights_out.

Keep the public API stable:
  candidate_forward(q_input, weight, weight_scale, freqs_cis, positions)

Success means candidate parity against the SGLang JIT path and better steady-state latency.
"""


task = restore_reference_parity.task(
    prompt=_PROMPT,
    reference_spec={
        "experiment": "dsv4_indexer.indexer",
        "reference": "sglang_forward",
        "candidate": "candidate_forward",
    },
    graders=[
        grader("check_dsv4_indexer_speed", args=W, weight=0.70, timeout=900, score_stdout=True),
        grader("check_dsv4_indexer_golden_speed", args=f"{W} {S}", weight=0.30, timeout=1800, score_stdout=True),
    ],
    patches=[],
    setup_command=f"cp -r {S}/tasks/sglang_dsv4_indexer/starter_files/dsv4_indexer {W}/dsv4_indexer && cp -r {S}/../sglang/python/sglang/jit_kernel {W}/sglang_jit_kernel",
)
task.slug = "sglang_dsv4_indexer"
