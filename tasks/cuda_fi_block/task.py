from env import WORKSPACE as W, restore_reference_parity
from tasks.graders import grader
from tasks.utils import load_patches


_PROMPT = """A TorchTitan DeepSeek-style inference block has a tuned V2 reference and a slow candidate path. Optimize the candidate implementation.

Target files:
  torchtitan/experiments/cuda_fusion_block/custom_ops.py
  torchtitan/experiments/cuda_fusion_block/cuda_ext/*
  torchtitan/experiments/cuda_fusion_block/deepseek_block.py only if needed for dispatch glue

Constraints:
  - Do not change eager_forward to make numbers easier.
  - Do not import or call FlashInfer from the candidate path; graders use a hidden tuned V2 golden reference for speed/parity.
  - You may use PyTorch GEMMs and PyTorch scaled_dot_product_attention.
  - The custom_ops public API must remain stable:
    - custom_ops.rmsnorm
    - custom_ops.fused_add_rmsnorm_
    - custom_ops.apply_rope_
    - custom_ops.silu_and_mul

Run locally:
  python torchtitan/experiments/cuda_fusion_block/benchmark_deepseek_block.py

Success means hidden parity passes against eager and the tuned V2 golden reference, the candidate uses a compiled CUDA extension, the candidate does not call FlashInfer, and steady-state block latency approaches the golden reference.
"""


task = restore_reference_parity.task(
    prompt=_PROMPT,
    reference_spec={
        "experiment": "cuda_fusion_block.deepseek_block",
        "reference": "eager_forward+golden_v2",
        "candidate": "candidate_forward",
        "forbidden_candidate_dependency": "flashinfer",
    },
    graders=[
        grader("check_cuda_block_extension", args=W, weight=0.20, timeout=300),
        grader("check_cuda_block_no_flashinfer", args=W, weight=0.15, timeout=120),
        grader("check_cuda_block_parity", args=W, weight=0.30, timeout=600),
        grader("check_cuda_block_speed", args=W, weight=0.35, timeout=900, score_stdout=True),
    ],
    patches=load_patches(__file__),
)
task.slug = "cuda_fi_block"
