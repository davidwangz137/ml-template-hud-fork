"""Candidate-owned fused ops for the DeepSeek block CUDA-fusion task."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    ext_dir = Path(__file__).with_name("cuda_ext")
    cu = ext_dir / "block_kernels.cu"
    cpp = ext_dir / "block_kernels.cpp"
    if not cu.exists() or not cpp.exists():
        _EXTENSION = False
        return None
    # Starter files intentionally contain this marker so the task begins with
    # correct PyTorch fallbacks instead of loading an empty extension.
    if "Stub file for the CUDA fusion block HUD task" in cu.read_text(errors="ignore") + cpp.read_text(errors="ignore"):
        _EXTENSION = False
        return None
    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="cuda_fusion_block_ops",
            sources=[str(cpp), str(cu)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        required = ("rmsnorm", "fused_add_rmsnorm_", "apply_rope_", "silu_and_mul")
        if any(not hasattr(_EXTENSION, name) for name in required):
            _EXTENSION = False
            return None
    except Exception:
        _EXTENSION = False
        return None
    return _EXTENSION


def _check_cuda_contig(name: str, tensor: torch.Tensor, ndim: int | None = None) -> None:
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {tuple(tensor.shape)}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _check_cuda_contig("input", input, 2)
    if weight.shape != (input.shape[1],):
        raise ValueError(f"weight shape must be ({input.shape[1]},), got {tuple(weight.shape)}")
    ext = _load_extension()
    if ext is not None and ext is not False:
        return ext.rmsnorm(input, weight, float(eps))
    return F.rms_norm(input.float(), (input.shape[-1],), weight.float(), eps).to(input.dtype)


def fused_add_rmsnorm_(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    _check_cuda_contig("input", input, 2)
    _check_cuda_contig("residual", residual, 2)
    if input.shape != residual.shape:
        raise ValueError(f"input/residual shape mismatch: {input.shape} vs {residual.shape}")
    if weight.shape != (input.shape[1],):
        raise ValueError(f"weight shape must be ({input.shape[1]},), got {tuple(weight.shape)}")
    ext = _load_extension()
    if ext is not None and ext is not False:
        ext.fused_add_rmsnorm_(input, residual, weight, float(eps))
        return
    residual.add_(input)
    input.copy_(F.rms_norm(residual.float(), (residual.shape[-1],), weight.float(), eps).to(input.dtype))


def apply_rope_(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    _check_cuda_contig("q", q, 4)
    _check_cuda_contig("k", k, 4)
    _check_cuda_contig("cos", cos, 2)
    _check_cuda_contig("sin", sin, 2)
    ext = _load_extension()
    if ext is not None and ext is not False:
        ext.apply_rope_(q, k, cos, sin)
        return
    half = q.shape[-1] // 2
    c = cos[None, :, None, :].float()
    s = sin[None, :, None, :].float()
    for tensor in (q, k):
        even = tensor[..., :half].float()
        odd = tensor[..., half:].float()
        tensor[..., :half].copy_((even * c - odd * s).to(tensor.dtype))
        tensor[..., half:].copy_((even * s + odd * c).to(tensor.dtype))


def silu_and_mul(gate_up: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    _check_cuda_contig("gate_up", gate_up, 2)
    if gate_up.shape[1] % 2 != 0:
        raise ValueError("gate_up second dimension must be even")
    hidden_dim = gate_up.shape[1] // 2
    if out is None:
        out = torch.empty((gate_up.shape[0], hidden_dim), device=gate_up.device, dtype=gate_up.dtype)
    else:
        _check_cuda_contig("out", out, 2)
        if out.shape != (gate_up.shape[0], hidden_dim):
            raise ValueError(f"out shape must be {(gate_up.shape[0], hidden_dim)}, got {tuple(out.shape)}")
    ext = _load_extension()
    if ext is not None and ext is not False:
        return ext.silu_and_mul(gate_up, out)
    out.copy_((F.silu(gate_up[:, :hidden_dim].float()) * gate_up[:, hidden_dim:].float()).to(gate_up.dtype))
    return out
