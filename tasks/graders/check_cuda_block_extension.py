"""Verify cuda_fi_block candidate uses a real CUDA extension."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    ws = Path(argv[0])
    root = ws / "torchtitan" / "experiments" / "cuda_fusion_block"
    ext = root / "cuda_ext"
    ops = root / "custom_ops.py"
    cu_files = sorted(ext.rglob("*.cu")) if ext.exists() else []
    cpp_files = sorted([*ext.rglob("*.cpp"), *ext.rglob("*.cc"), *ext.rglob("*.cxx")]) if ext.exists() else []
    if not cu_files or not cpp_files or not ops.exists():
        print("missing CUDA extension files")
        return 1
    cu_text = "\n".join(p.read_text(errors="replace") for p in cu_files)
    cpp_text = "\n".join(p.read_text(errors="replace") for p in cpp_files)
    combined = cu_text + "\n" + cpp_text + "\n" + ops.read_text(errors="replace")
    if "Stub file for the CUDA fusion block HUD task" in cu_text + cpp_text:
        print("stub extension still present")
        return 1
    for snippet in ("__global__", "PYBIND11_MODULE"):
        if snippet not in combined:
            print(f"missing {snippet}")
            return 1
    for forbidden in ("import flashinfer", "from flashinfer", "torch.ops.flashinfer"):
        if forbidden in combined:
            print(f"candidate references FlashInfer: {forbidden}")
            return 1
    sys.path.insert(0, str(ws))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ws / ".torch_extensions"))
    try:
        import torch
        from torchtitan.experiments.cuda_fusion_block.deepseek_block import DeepSeekBlockShape, candidate_forward, clone_case, make_inputs
        x, weights, cos, sin = make_inputs(DeepSeekBlockShape(seq=33, dim=256, heads=4, kv_heads=2, head_dim=64, ffn_hidden_dim=512), seed=33)
        out = candidate_forward(*clone_case(x, weights, cos, sin))
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            print("candidate produced non-finite output")
            return 1
    except Exception as exc:
        print(f"extension smoke failed: {type(exc).__name__}: {exc}")
        return 1
    print("block extension passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
