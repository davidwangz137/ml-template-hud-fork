"""Hidden parity checks for cuda_fi_block."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def metrics(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    ws = Path(argv[0])
    sys.path.insert(0, str(ws))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ws / ".torch_extensions"))
    try:
        import torch
        from torchtitan.experiments.cuda_fusion_block.deepseek_block import DeepSeekBlockShape, candidate_forward, clone_case, compiler_forward, eager_forward, make_inputs
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}")
        return 1
    shapes = [
        DeepSeekBlockShape(seq=64, dim=256, heads=4, kv_heads=2, head_dim=64, ffn_hidden_dim=512),
        DeepSeekBlockShape(seq=129, dim=512, heads=8, kv_heads=2, head_dim=64, ffn_hidden_dim=1536),
        DeepSeekBlockShape(seq=257, dim=768, heads=12, kv_heads=3, head_dim=64, ffn_hidden_dim=2048),
    ]
    for i, shape in enumerate(shapes):
        x, weights, cos, sin = make_inputs(shape, seed=18000 + i)
        with torch.no_grad():
            eager = eager_forward(*clone_case(x, weights, cos, sin))
            compiler = compiler_forward(*clone_case(x, weights, cos, sin))
            cand = candidate_forward(*clone_case(x, weights, cos, sin))
        torch.cuda.synchronize()
        if not torch.isfinite(cand).all():
            print(f"non-finite candidate for {shape}")
            return 1
        max_e, mean_e = metrics(cand, eager)
        max_c, mean_c = metrics(cand, compiler)
        print(f"shape={shape} max_eager={max_e:.6g} mean_eager={mean_e:.6g} max_compiler={max_c:.6g} mean_compiler={mean_c:.6g}")
        if max_e > 0.12 or mean_e > 0.02 or max_c > 0.12 or mean_c > 0.02:
            print("parity failed")
            return 1
    print("block parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
