#!/usr/bin/env python3
"""Benchmark the SGLang DeepSeek-V4 indexer subgraph.

This is a small, real SGLang-serving kernel chain extracted from the DSV4 JIT
path, not the older V2 DeepSeek block:

    q_input[B, H, 128] bf16/fp16
      -> RoPE on the final 64 channels
      -> 128-point Hadamard transform
      -> per-(B,H) FP8 E4M3 activation quantization
      -> weights_out = weight * weight_scale * activation_scale

Versions:

* eager: PyTorch reference for the same dataflow and layout.
* compiler: torch.compile(eager), when supported by the installed PyTorch.
* sglang: sglang.jit_kernel.dsv4.elementwise.fused_q_indexer_rope_hadamard_quant.
* candidate: optional module:function with the same call contract as eager.

Candidate contract:

    def candidate(q_input, weight, weight_scale, freqs_cis, positions) -> tuple[torch.Tensor, torch.Tensor]

Return ``(q_fp8, weights_out)`` where q_fp8 has dtype ``torch.float8_e4m3fn``
and shape ``[B, H, 128]``, while weights_out has shape ``[B, H, 1]`` fp32.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SGLANG_KERNEL = (REPO_ROOT / ".." / "sglang" / "python" / "sglang" / "jit_kernel").resolve()
_COMPILED_EAGER: Callable | None = None
FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class DSV4IndexerShape:
    batch: int = 64
    heads: int = 16
    head_dim: int = 128
    rope_dim: int = 64
    max_seq_len: int = 8192
    dtype: str = "bfloat16"

    def validate(self) -> None:
        if self.batch <= 0 or self.heads <= 0:
            raise ValueError("batch and heads must be positive")
        if self.head_dim != 128:
            raise ValueError("SGLang DSV4 indexer requires head_dim=128")
        if self.rope_dim != 64:
            raise ValueError("SGLang DSV4 indexer requires rope_dim=64")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        raise ValueError(f"unsupported dtype {self.dtype!r}; use bfloat16 or float16")


def _prepend_python_bin_to_path() -> None:
    candidate_bins = [Path(sys.prefix).resolve() / "bin", Path(sys.executable).resolve().parent]
    old_path = os.environ.get("PATH", "")
    path_entries = old_path.split(os.pathsep) if old_path else []
    new_entries = [str(path) for path in candidate_bins if path.exists() and str(path) not in path_entries]
    if new_entries:
        os.environ["PATH"] = os.pathsep.join(new_entries + ([old_path] if old_path else []))


def _patch_flashinfer_ninja() -> None:
    """Use the venv ninja binary for TVM/FlashInfer/SGLang JIT subprocesses."""
    ninja = shutil.which("ninja")
    if ninja is None:
        venv_ninja = Path(sys.prefix).resolve() / "bin" / "ninja"
        if not venv_ninja.exists():
            return
        ninja = str(venv_ninja)
    try:
        import flashinfer.jit.core as flashinfer_core
        import flashinfer.jit.cpp_ext as flashinfer_cpp_ext
    except ModuleNotFoundError:
        return

    def run_ninja_with_absolute_binary(workdir: Path, ninja_file: Path, verbose: bool) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        command = [ninja, "-v", "-C", str(workdir.resolve()), "-f", str(ninja_file.resolve())]
        max_jobs = os.environ.get("MAX_JOBS")
        if max_jobs is not None and max_jobs.isdigit():
            command += ["-j", max_jobs]
        completed = subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            msg = "Ninja build failed."
            if completed.stdout:
                msg += " Ninja output:\n" + completed.stdout
            raise RuntimeError(msg)

    flashinfer_cpp_ext.run_ninja = run_ninja_with_absolute_binary
    flashinfer_core.run_ninja = run_ninja_with_absolute_binary


def _load_sglang_dsv4_elementwise():
    """Load SGLang DSV4 JIT modules without importing sglang.__init__."""
    kernel_root = LOCAL_SGLANG_KERNEL
    if not kernel_root.exists():
        raise ImportError(f"SGLang JIT kernel checkout not found at {kernel_root}")

    def package(name: str, path: Path) -> None:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    package("sglang", kernel_root.parent)
    package("sglang.jit_kernel", kernel_root)
    package("sglang.jit_kernel.dsv4", kernel_root / "dsv4")
    package("sglang.srt", kernel_root.parent / "srt")

    srt_utils = sys.modules.get("sglang.srt.utils")
    if srt_utils is None:
        srt_utils = types.ModuleType("sglang.srt.utils")
        srt_utils.is_hip = lambda: False
        sys.modules["sglang.srt.utils"] = srt_utils

    sglang_utils = sys.modules.get("sglang.utils")
    if sglang_utils is None:
        sglang_utils = types.ModuleType("sglang.utils")
        sglang_utils.is_in_ci = lambda: False
        sys.modules["sglang.utils"] = sglang_utils

    for name, path in (
        ("sglang.jit_kernel.utils", kernel_root / "utils.py"),
        ("sglang.jit_kernel.dsv4.utils", kernel_root / "dsv4" / "utils.py"),
        ("sglang.jit_kernel.dsv4.elementwise", kernel_root / "dsv4" / "elementwise.py"),
    ):
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["sglang.jit_kernel.dsv4.elementwise"]


def _make_freqs_cis(max_seq_len: int, rope_dim: int, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles).contiguous()


def make_inputs(shape: DSV4IndexerShape, seed: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
    shape.validate()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q_input = torch.randn(shape.batch, shape.heads, shape.head_dim, device=device, dtype=shape.torch_dtype, generator=gen)
    weight = torch.randn(shape.batch, shape.heads, device=device, dtype=shape.torch_dtype, generator=gen)
    weight_scale = 0.03125
    freqs_cis = _make_freqs_cis(shape.max_seq_len, shape.rope_dim, device)
    positions = torch.randint(0, shape.max_seq_len, (shape.batch,), device=device, dtype=torch.int32, generator=gen)
    return q_input, weight, weight_scale, freqs_cis, positions


def _rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rope_dim = 64
    out = x.clone()
    tail = out[..., -rope_dim:].float().reshape(*out.shape[:-1], rope_dim // 2, 2)
    freqs = torch.view_as_real(freqs_cis[positions]).view(x.shape[0], 1, rope_dim // 2, 2)
    real = tail[..., 0] * freqs[..., 0] - tail[..., 1] * freqs[..., 1]
    imag = tail[..., 0] * freqs[..., 1] + tail[..., 1] * freqs[..., 0]
    out[..., -rope_dim:] = torch.stack((real, imag), dim=-1).reshape(*out.shape[:-1], rope_dim).to(out.dtype)
    return out


def _sglang_hadamard_128_order(x: torch.Tensor) -> torch.Tensor:
    """Match the lane-major Hadamard ordering in main_norm_rope.cuh."""
    data = x.float().view(*x.shape[:-1], 32, 4)
    a0, a1, a2, a3 = data.unbind(dim=-1)
    local = torch.stack((a0 + a1, a0 - a1, a2 + a3, a2 - a3), dim=-1)
    b0, b1, b2, b3 = local.unbind(dim=-1)
    data = torch.stack((b0 + b2, b1 + b3, b0 - b2, b1 - b3), dim=-1)
    # Five cross-lane shfl_xor stages over the 32-lane dimension. The CUDA
    # branch uses (lane_id & mask) ? other - data : data + other.
    for mask in (1, 2, 4, 8, 16):
        idx = torch.arange(32, device=x.device)
        other = data.index_select(-2, idx ^ mask)
        sign = torch.where((idx & mask) == 0, 1.0, -1.0).view(*([1] * (data.ndim - 2)), 32, 1)
        data = other + sign * data
    return (data * (1.0 / (128.0 ** 0.5))).reshape_as(x)


def eager_forward(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    roped = _rope_tail(q_input, freqs_cis, positions)
    hadamard = _sglang_hadamard_128_order(roped)
    abs_max = hadamard.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(abs_max, min=1e-4) / FP8_E4M3_MAX
    q_fp8 = (hadamard / scale).to(torch.float8_e4m3fn)
    weights_out = weight.float().unsqueeze(-1) * float(weight_scale) * scale
    return q_fp8, weights_out


def compiler_forward(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _COMPILED_EAGER
    if _COMPILED_EAGER is None:
        _COMPILED_EAGER = torch.compile(eager_forward, mode="max-autotune", fullgraph=False)
    return _COMPILED_EAGER(q_input, weight, weight_scale, freqs_cis, positions)


def sglang_forward(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _prepend_python_bin_to_path()
    _patch_flashinfer_ninja()
    fused = _load_sglang_dsv4_elementwise().fused_q_indexer_rope_hadamard_quant
    return fused(q_input.contiguous(), weight.contiguous(), float(weight_scale), freqs_cis, positions)


def load_candidate(path: str | None) -> Callable | None:
    if not path:
        return None
    if ":" not in path:
        raise ValueError("--candidate must be module:function")
    module_name, function_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"candidate {path!r} is not callable")
    return fn


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def _output_stats(lhs: tuple[torch.Tensor, torch.Tensor], rhs: tuple[torch.Tensor, torch.Tensor], prefix: str) -> dict[str, float]:
    lhs_q, lhs_w = lhs
    rhs_q, rhs_w = rhs
    return {
        f"{prefix}_q_fp8_as_float_max_abs": _max_abs(lhs_q, rhs_q),
        f"{prefix}_q_fp8_as_float_mean_abs": _mean_abs(lhs_q, rhs_q),
        f"{prefix}_weights_out_max_abs": _max_abs(lhs_w, rhs_w),
        f"{prefix}_weights_out_mean_abs": _mean_abs(lhs_w, rhs_w),
    }


def _time(fn: Callable, q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(q_input, weight, weight_scale, freqs_cis, positions)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn(q_input, weight, weight_scale, freqs_cis, positions)
        end.record()
        torch.cuda.synchronize()
    if not torch.isfinite(out[1]).all():
        raise RuntimeError("non-finite weights_out")
    return start.elapsed_time(end) / iters


def _shape(spec: str) -> DSV4IndexerShape:
    batch, heads, max_seq_len = (int(x) for x in spec.split(","))
    return DSV4IndexerShape(batch=batch, heads=heads, max_seq_len=max_seq_len)


def run_one(shape: DSV4IndexerShape, seed: int, warmup: int, iters: int, repeats: int, candidate: Callable | None) -> dict[str, Any]:
    q_input, weight, weight_scale, freqs_cis, positions = make_inputs(shape, seed)
    versions: dict[str, Callable] = {
        "eager": eager_forward,
        "compiler": compiler_forward,
        "sglang": sglang_forward,
    }
    if candidate is not None:
        versions["candidate"] = candidate

    with torch.no_grad():
        outputs = {name: fn(q_input, weight, weight_scale, freqs_cis, positions) for name, fn in versions.items()}
    eager = outputs["eager"]

    timings: dict[str, list[float]] = {name: [] for name in versions}
    for rep in range(repeats):
        rq, rw, rws, rf, rp = make_inputs(shape, seed + 4099 * (rep + 1))
        for name, fn in versions.items():
            timings[name].append(_time(fn, rq, rw, rws, rf, rp, warmup, iters))

    medians = {f"{name}_ms": statistics.median(values) for name, values in timings.items()}
    result: dict[str, Any] = {
        "shape": asdict(shape),
        **medians,
        **_output_stats(outputs["compiler"], eager, "compiler_vs_eager"),
        **_output_stats(outputs["sglang"], eager, "sglang_vs_eager"),
        "sglang_vs_eager_speedup": medians["eager_ms"] / medians["sglang_ms"],
        "sglang_vs_compiler_speedup": medians["compiler_ms"] / medians["sglang_ms"],
    }
    if candidate is not None:
        result.update(
            {
                **_output_stats(outputs["candidate"], eager, "candidate_vs_eager"),
                **_output_stats(outputs["candidate"], outputs["sglang"], "candidate_vs_sglang"),
                "candidate_vs_eager_speedup": medians["eager_ms"] / medians["candidate_ms"],
                "candidate_vs_compiler_speedup": medians["compiler_ms"] / medians["candidate_ms"],
                "candidate_vs_sglang_speedup": medians["sglang_ms"] / medians["candidate_ms"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", default=None, help="batch,heads,max_seq_len; fixed head_dim=128 rope_dim=64")
    parser.add_argument("--candidate", default=None, help="Optional candidate module:function")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    shapes = [_shape(s) for s in args.shape] if args.shape else [
        DSV4IndexerShape(batch=64, heads=16, max_seq_len=8192),
        DSV4IndexerShape(batch=256, heads=16, max_seq_len=8192),
    ]
    candidate = load_candidate(args.candidate)
    results = [run_one(shape, args.seed + idx, args.warmup, args.iters, args.repeats, candidate) for idx, shape in enumerate(shapes)]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
