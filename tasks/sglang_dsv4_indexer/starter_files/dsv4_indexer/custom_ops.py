"""Candidate-owned CUDA ops for the DSV4 indexer subgraph."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import torch

_EXTENSION = None


def _prepend_python_bin_to_path() -> None:
    python_bin = Path(sys.prefix).resolve() / "bin"
    old_path = os.environ.get("PATH", "")
    entries = old_path.split(os.pathsep) if old_path else []
    python_bin_s = str(python_bin)
    if python_bin.exists() and python_bin_s not in entries:
        os.environ["PATH"] = python_bin_s + (os.pathsep + old_path if old_path else "")


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    ext_dir = Path(__file__).with_name("cuda_ext")
    cu = ext_dir / "indexer_kernels.cu"
    cpp = ext_dir / "indexer_kernels.cpp"
    if not cu.exists() or not cpp.exists():
        _EXTENSION = False
        return None
    _prepend_python_bin_to_path()
    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="dsv4_indexer_kernels",
            sources=[str(cpp), str(cu)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _EXTENSION = False
        if torch.cuda.is_available():
            raise RuntimeError("Failed to build/load DSV4 indexer CUDA extension") from exc
        return None
    return _EXTENSION


def indexer(q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if q_input.ndim != 3 or q_input.shape[-1] != 128:
        raise ValueError("q_input must have shape [B, H, 128]")
    if weight.shape != q_input.shape[:2]:
        raise ValueError("weight must have shape [B, H]")
    if positions.shape != (q_input.shape[0],):
        raise ValueError("positions must have shape [B]")
    if q_input.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q_input must be bf16 or fp16")
    if not q_input.is_cuda:
        from .indexer import eager_forward

        return eager_forward(q_input, weight, weight_scale, freqs_cis, positions)

    ext = _load_extension()
    if ext is None:
        from .indexer import eager_forward

        return eager_forward(q_input, weight, weight_scale, freqs_cis, positions)

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    return ext.indexer(q_input.contiguous(), weight.contiguous(), float(weight_scale), freqs_real, positions.contiguous())
