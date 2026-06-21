"""Verify cuda_fi_block candidate does not call FlashInfer."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


class FlashInferSentinel(types.ModuleType):
    def __init__(self):
        super().__init__("flashinfer")
        self.__file__ = "<blocked>"
        self.__path__ = []
        self.__spec__ = None

    def __getattr__(self, name: str):
        raise RuntimeError(f"candidate accessed flashinfer.{name}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    ws = Path(argv[0])
    root = ws / "torchtitan" / "experiments" / "cuda_fusion_block"
    paths = [root / "custom_ops.py"] + ([p for p in (root / "cuda_ext").rglob("*") if p.is_file()] if (root / "cuda_ext").exists() else [])
    for p in paths:
        text = p.read_text(errors="replace")
        for forbidden in ("import flashinfer", "from flashinfer", "torch.ops.flashinfer", "flashinfer::"):
            if forbidden in text:
                print(f"{p.relative_to(ws)} references {forbidden}")
                return 1
    sys.path.insert(0, str(ws))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ws / ".torch_extensions"))
    sys.modules["flashinfer"] = FlashInferSentinel()
    try:
        import torch
        from torchtitan.experiments.cuda_fusion_block.deepseek_block import DeepSeekBlockShape, candidate_forward, clone_case, make_inputs
        x, weights, cos, sin = make_inputs(DeepSeekBlockShape(seq=35, dim=256, heads=4, kv_heads=2, head_dim=64, ffn_hidden_dim=512), seed=35)
        out = candidate_forward(*clone_case(x, weights, cos, sin))
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            print("candidate produced non-finite output")
            return 1
    except Exception as exc:
        print(f"candidate failed with FlashInfer blocked: {type(exc).__name__}: {exc}")
        return 1
    print("block candidate runs with FlashInfer blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
