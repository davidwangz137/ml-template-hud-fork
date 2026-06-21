"""Speed checks for cuda_fi_block."""

from __future__ import annotations

import os
import statistics
import math
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    ws = Path(argv[0])
    root = ws / "torchtitan" / "experiments" / "cuda_fusion_block" / "cuda_ext"
    text = "\n".join(p.read_text(errors="replace") for p in root.rglob("*") if p.is_file()) if root.exists() else ""
    if "Stub file for the CUDA fusion block HUD task" in text or "__global__" not in text:
        print("speed check requires real CUDA source")
        return 1
    sys.path.insert(0, str(ws))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ws / ".torch_extensions"))
    try:
        import torch
        from torchtitan.experiments.cuda_fusion_block.deepseek_block import DeepSeekBlockShape, candidate_forward, clone_case, compiler_forward, eager_forward, make_inputs
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}")
        return 1

    def time_fn(fn, x, weights, cos, sin, warmup=5, iters=20):
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

    shapes = [
        DeepSeekBlockShape(seq=257, dim=768, heads=12, kv_heads=3, head_dim=64, ffn_hidden_dim=2048),
        DeepSeekBlockShape(seq=384, dim=768, heads=12, kv_heads=3, head_dim=64, ffn_hidden_dim=2048),
    ]
    speedups = []
    compiler_ratios = []
    for i, shape in enumerate(shapes):
        x, weights, cos, sin = make_inputs(shape, seed=22000 + i)
        eager = statistics.median(time_fn(eager_forward, x, weights, cos, sin) for _ in range(3))
        compiler = statistics.median(time_fn(compiler_forward, x, weights, cos, sin) for _ in range(3))
        cand = statistics.median(time_fn(candidate_forward, x, weights, cos, sin) for _ in range(3))
        cand_vs_eager = eager / cand if cand > 0 else 0.0
        cand_vs_compiler = compiler / cand if cand > 0 else 0.0
        speedups.append(cand_vs_eager)
        compiler_ratios.append(cand_vs_compiler)
        print(f"shape={shape} eager_ms={eager:.4f} compiler_ms={compiler:.4f} candidate_ms={cand:.4f} cand_vs_eager={cand_vs_eager:.3f} cand_vs_compiler={cand_vs_compiler:.3f}")
        if cand_vs_eager > 50.0:
            print(f"untrusted speedup over 50x for {shape}; possible reward hacking")
            return 1

    geomean_speedup = math.prod(max(s, 1e-6) for s in speedups) ** (1.0 / len(speedups))
    geomean_vs_compiler = math.prod(max(s, 1e-6) for s in compiler_ratios) ** (1.0 / len(compiler_ratios))
    # Continuous reward: 1.0 at 5x geomean speedup over eager, clamped.
    # Better-than-reference solutions earn more until the cap; correctness is
    # enforced by separate parity/no-FlashInfer/provenance graders.
    score = max(0.0, min(1.0, geomean_speedup / 5.0))
    print(f"geomean_speedup_vs_eager={geomean_speedup:.4f} geomean_vs_compiler={geomean_vs_compiler:.4f}")
    print(f"SCORE: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
