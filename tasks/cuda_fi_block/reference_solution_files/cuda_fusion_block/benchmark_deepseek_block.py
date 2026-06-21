"""Profile a DeepSeek-style single decoder block."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

try:
    from torchtitan.experiments.cuda_fusion_block.deepseek_block import (
        DeepSeekBlockShape,
        candidate_forward,
        clone_case,
        eager_forward,
        make_inputs,
    )
except ModuleNotFoundError:
    import os
    import sys

    sys.path.insert(0, os.getcwd())
    from torchtitan.experiments.cuda_fusion_block.deepseek_block import (
        DeepSeekBlockShape,
        candidate_forward,
        clone_case,
        eager_forward,
        make_inputs,
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _time(fn: Callable, x, weights, cos, sin, warmup: int, iters: int) -> float:
    """Time inference with static weights.

    Clone only the mutable activation input. Weights/RoPE tables are immutable
    inference state; cloning them inside the timed region measures allocator and
    memory-copy overhead, not block execution.
    """
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(x.clone(), weights, cos, sin)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn(x.clone(), weights, cos, sin)
        end.record()
        torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite output")
    return start.elapsed_time(end) / iters


def _shape(spec: str) -> DeepSeekBlockShape:
    seq, dim, heads, kv_heads, head_dim, ffn = (int(x) for x in spec.split(","))
    return DeepSeekBlockShape(seq=seq, dim=dim, heads=heads, kv_heads=kv_heads, head_dim=head_dim, ffn_hidden_dim=ffn)


def run_one(shape: DeepSeekBlockShape, seed: int, warmup: int, iters: int, repeats: int) -> dict:
    x, weights, cos, sin = make_inputs(shape, seed)
    with torch.no_grad():
        eager = eager_forward(*clone_case(x, weights, cos, sin))
        candidate = candidate_forward(*clone_case(x, weights, cos, sin))
    eager_times = []
    candidate_times = []
    for rep in range(repeats):
        rx, rw, rc, rs = make_inputs(shape, seed + 4099 * (rep + 1))
        eager_times.append(_time(eager_forward, rx, rw, rc, rs, warmup, iters))
        candidate_times.append(_time(candidate_forward, rx, rw, rc, rs, warmup, iters))
    eager_ms = statistics.median(eager_times)
    candidate_ms = statistics.median(candidate_times)
    return {
        "shape": {
            "seq": shape.seq,
            "dim": shape.dim,
            "heads": shape.heads,
            "kv_heads": shape.kv_heads,
            "head_dim": shape.head_dim,
            "ffn_hidden_dim": shape.ffn_hidden_dim,
            "dtype": str(shape.dtype),
        },
        "max_abs_vs_eager": _max_abs(candidate, eager),
        "mean_abs_vs_eager": _mean_abs(candidate, eager),
        "candidate_ms": candidate_ms,
        "candidate_vs_eager": eager_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", default=None, help="seq,dim,heads,kv_heads,head_dim,ffn")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    shapes = [_shape(s) for s in args.shape] if args.shape else [
        DeepSeekBlockShape(seq=128, dim=512, heads=8, kv_heads=2, head_dim=64, ffn_hidden_dim=2048),
        DeepSeekBlockShape(seq=257, dim=768, heads=12, kv_heads=3, head_dim=64, ffn_hidden_dim=3072),
    ]
    results = [run_one(shape, args.seed + idx, args.warmup, args.iters, args.repeats) for idx, shape in enumerate(shapes)]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
