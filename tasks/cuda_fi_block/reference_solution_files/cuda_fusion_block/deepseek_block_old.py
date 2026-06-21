"""DeepSeek-style single decoder block for profiling-driven CUDA fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import os

from . import custom_ops

_COMPILED_COMPILER = None
_COMPILED_CANDIDATE = None
_COMPILE_COMPILER_FAILED = False
_COMPILE_CANDIDATE_FAILED = False


def _use_torch_compile() -> bool:
    return not bool(int(os.environ.get("CUDA_FI_BLOCK_DISABLE_COMPILE", "0")))


@dataclass(frozen=True)
class DeepSeekBlockShape:
    seq: int = 128
    dim: int = 512
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden_dim: int = 2048
    dtype: torch.dtype = torch.bfloat16

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class DeepSeekBlockWeights:
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    w_qkv: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    w_gate_up: torch.Tensor


def make_rope(seq: int, head_dim: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    pos = torch.arange(seq, device=device, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(half, device=device, dtype=torch.float32)[None, :] / half))
    freqs = pos * inv
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


def make_inputs(shape: DeepSeekBlockShape, seed: int, device: str = "cuda") -> tuple[torch.Tensor, DeepSeekBlockWeights, torch.Tensor, torch.Tensor]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(shape.seq, shape.dim, device=device, dtype=shape.dtype, generator=g)
    scale = 0.02
    w_q = torch.randn(shape.dim, shape.q_dim, device=device, dtype=shape.dtype, generator=g) * scale
    w_k = torch.randn(shape.dim, shape.kv_dim, device=device, dtype=shape.dtype, generator=g) * scale
    w_v = torch.randn(shape.dim, shape.kv_dim, device=device, dtype=shape.dtype, generator=g) * scale
    w_gate = torch.randn(shape.dim, shape.ffn_hidden_dim, device=device, dtype=shape.dtype, generator=g) * scale
    w_up = torch.randn(shape.dim, shape.ffn_hidden_dim, device=device, dtype=shape.dtype, generator=g) * scale
    weights = DeepSeekBlockWeights(
        attn_norm_weight=torch.ones(shape.dim, device=device, dtype=shape.dtype),
        ffn_norm_weight=torch.ones(shape.dim, device=device, dtype=shape.dtype),
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_qkv=torch.cat((w_q, w_k, w_v), dim=1).contiguous(),
        w_o=torch.randn(shape.q_dim, shape.dim, device=device, dtype=shape.dtype, generator=g) * scale,
        w_gate=w_gate,
        w_up=w_up,
        w_gate_up=torch.cat((w_gate, w_up), dim=1).contiguous(),
        w_down=torch.randn(shape.ffn_hidden_dim, shape.dim, device=device, dtype=shape.dtype, generator=g) * scale,
    )
    cos, sin = make_rope(shape.seq, shape.head_dim, device=device)
    return x, weights, cos, sin


def clone_case(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor):
    return x.clone(), DeepSeekBlockWeights(*(t.clone() for t in weights.__dict__.values())), cos.clone(), sin.clone()


def _reshape_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: DeepSeekBlockShape):
    q = q.view(1, shape.seq, shape.heads, shape.head_dim).contiguous()
    k = k.view(1, shape.seq, shape.kv_heads, shape.head_dim).contiguous()
    v = v.view(1, shape.seq, shape.kv_heads, shape.head_dim).contiguous()
    return q, k, v


def _split_qkv(qkv: torch.Tensor, shape: DeepSeekBlockShape):
    q_end = shape.q_dim
    k_end = q_end + shape.kv_dim
    return _reshape_qkv(qkv[:, :q_end], qkv[:, q_end:k_end], qkv[:, k_end:], shape)


def _rope_eager(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    half = q.shape[-1] // 2
    c = cos[None, :, None, :].float()
    s = sin[None, :, None, :].float()
    for tensor in (q, k):
        a = tensor[..., :half].float()
        b = tensor[..., half:].float()
        tensor[..., :half].copy_((a * c - b * s).to(tensor.dtype))
        tensor[..., half:].copy_((a * s + b * c).to(tensor.dtype))


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, enable_gqa: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
        enable_gqa=enable_gqa,
    ).permute(0, 2, 1, 3).contiguous().view(q.shape[1], -1)


def eager_forward(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = DeepSeekBlockShape(seq=x.shape[0], dim=x.shape[1], heads=weights.w_q.shape[1] // cos.shape[1] // 2, kv_heads=weights.w_k.shape[1] // cos.shape[1] // 2, head_dim=cos.shape[1] * 2, ffn_hidden_dim=weights.w_gate.shape[1], dtype=x.dtype)
    normed = F.rms_norm(x.float(), (shape.dim,), weights.attn_norm_weight.float(), eps).to(x.dtype)
    q, k, v = _reshape_qkv(normed @ weights.w_q, normed @ weights.w_k, normed @ weights.w_v, shape)
    _rope_eager(q, k, cos, sin)
    attn = _sdpa(q, k, v, shape.heads != shape.kv_heads)
    x = x + attn @ weights.w_o
    normed = F.rms_norm(x.float(), (shape.dim,), weights.ffn_norm_weight.float(), eps).to(x.dtype)
    gate = normed @ weights.w_gate
    up = normed @ weights.w_up
    hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
    return x + hidden @ weights.w_down

def _compiler_impl(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    import flashinfer

    shape = DeepSeekBlockShape(seq=x.shape[0], dim=x.shape[1], heads=weights.w_q.shape[1] // cos.shape[1] // 2, kv_heads=weights.w_k.shape[1] // cos.shape[1] // 2, head_dim=cos.shape[1] * 2, ffn_hidden_dim=weights.w_gate.shape[1], dtype=x.dtype)
    normed = F.rms_norm(x.float(), (shape.dim,), weights.attn_norm_weight.float(), eps).to(x.dtype)
    q, k, v = _split_qkv(normed @ weights.w_qkv, shape)
    _rope_eager(q, k, cos, sin)
    attn = flashinfer.single_prefill_with_kv_cache(q[0], k[0], v[0], causal=True, kv_layout="NHD").reshape(shape.seq, -1)
    attn_out = attn @ weights.w_o
    normed = attn_out.contiguous()
    residual = x.contiguous()
    flashinfer.fused_add_rmsnorm(normed, residual, weights.ffn_norm_weight, eps)
    gate_up = (normed @ weights.w_gate_up).contiguous()
    hidden = flashinfer.silu_and_mul(gate_up)
    return residual + hidden @ weights.w_down


def compiler_forward(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    global _COMPILED_COMPILER, _COMPILE_COMPILER_FAILED
    if not _use_torch_compile() or _COMPILE_COMPILER_FAILED:
        return _compiler_impl(x, weights, cos, sin, eps)
    if _COMPILED_COMPILER is None:
        try:
            _COMPILED_COMPILER = torch.compile(_compiler_impl, mode="reduce-overhead", fullgraph=False)
        except Exception:
            _COMPILE_COMPILER_FAILED = True
            return _compiler_impl(x, weights, cos, sin, eps)
    try:
        torch.compiler.cudagraph_mark_step_begin()
        return _COMPILED_COMPILER(x, weights, cos, sin, eps).clone()
    except Exception:
        _COMPILE_COMPILER_FAILED = True
        return _compiler_impl(x, weights, cos, sin, eps)
    gate_up = (normed @ weights.w_gate_up).contiguous()
    hidden = flashinfer.silu_and_mul(gate_up)
    return residual + hidden @ weights.w_down


def _candidate_impl(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    shape = DeepSeekBlockShape(seq=x.shape[0], dim=x.shape[1], heads=weights.w_q.shape[1] // cos.shape[1] // 2, kv_heads=weights.w_k.shape[1] // cos.shape[1] // 2, head_dim=cos.shape[1] * 2, ffn_hidden_dim=weights.w_gate.shape[1], dtype=x.dtype)
    normed = custom_ops.rmsnorm(x.contiguous(), weights.attn_norm_weight, eps)
    q, k, v = _split_qkv(normed @ weights.w_qkv, shape)
    custom_ops.apply_rope_(q, k, cos, sin)
    attn = _sdpa(q, k, v, shape.heads != shape.kv_heads)
    attn_out = attn @ weights.w_o
    residual = x + attn_out
    normed = F.rms_norm(residual.float(), (shape.dim,), weights.ffn_norm_weight.float(), eps).to(x.dtype)
    gate_up = (normed @ weights.w_gate_up).contiguous()
    hidden = custom_ops.silu_and_mul(gate_up)
    return residual + hidden @ weights.w_down


def candidate_forward(x: torch.Tensor, weights: DeepSeekBlockWeights, cos: torch.Tensor, sin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    global _COMPILED_CANDIDATE, _COMPILE_CANDIDATE_FAILED
    if not _use_torch_compile() or _COMPILE_CANDIDATE_FAILED:
        return _candidate_impl(x, weights, cos, sin, eps)
    if _COMPILED_CANDIDATE is None:
        try:
            _COMPILED_CANDIDATE = torch.compile(_candidate_impl, mode="reduce-overhead", fullgraph=False)
        except Exception:
            _COMPILE_CANDIDATE_FAILED = True
            return _candidate_impl(x, weights, cos, sin, eps)
    try:
        torch.compiler.cudagraph_mark_step_begin()
        return _COMPILED_CANDIDATE(x, weights, cos, sin, eps).clone()
    except Exception:
        _COMPILE_CANDIDATE_FAILED = True
        return _candidate_impl(x, weights, cos, sin, eps)
