"""DSV4 indexer subgraph: eager, compiler, SGLang, and CUDA candidate paths."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from . import custom_ops

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
        raise ValueError(f"unsupported dtype {self.dtype!r}")


def _prepend_python_bin_to_path() -> None:
    python_bin = Path(sys.prefix).resolve() / "bin"
    old_path = os.environ.get("PATH", "")
    path_entries = old_path.split(os.pathsep) if old_path else []
    python_bin_s = str(python_bin)
    if python_bin.exists() and python_bin_s not in path_entries:
        os.environ["PATH"] = python_bin_s + (os.pathsep + old_path if old_path else "")


def _patch_flashinfer_ninja() -> None:
    ninja = shutil.which("ninja") or str(Path(sys.prefix).resolve() / "bin" / "ninja")
    if not Path(ninja).exists():
        return
    try:
        import flashinfer.jit.core as flashinfer_core
        import flashinfer.jit.cpp_ext as flashinfer_cpp_ext
    except ModuleNotFoundError:
        return

    def run_ninja_with_absolute_binary(workdir: Path, ninja_file: Path, verbose: bool) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        command = [ninja, "-v", "-C", str(workdir.resolve()), "-f", str(ninja_file.resolve())]
        completed = subprocess.run(command, stdout=None if verbose else subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(workdir.resolve()), check=False, text=True)
        if completed.returncode != 0:
            msg = "Ninja build failed."
            if completed.stdout:
                msg += " Ninja output:\n" + completed.stdout
            raise RuntimeError(msg)

    flashinfer_cpp_ext.run_ninja = run_ninja_with_absolute_binary
    flashinfer_core.run_ninja = run_ninja_with_absolute_binary


def _load_sglang_dsv4_elementwise():
    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        candidates.extend(
            [
                parent / "sglang_jit_kernel",
                parent / "sglang" / "python" / "sglang" / "jit_kernel",
            ]
        )
    candidates.append(Path("/home/wangd/code_hackathon/sglang/python/sglang/jit_kernel"))
    kernel_root = next((path.resolve() for path in candidates if path.exists()), None)
    if kernel_root is None:
        raise ImportError("Could not find SGLang jit_kernel; expected staged sglang_jit_kernel in workspace")

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


def make_freqs_cis(max_seq_len: int, rope_dim: int, device: str) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    return torch.polar(torch.ones(max_seq_len, rope_dim // 2, device=device), torch.outer(pos, inv_freq)).contiguous()


def make_inputs(shape: DSV4IndexerShape, seed: int, device: str = "cuda"):
    shape.validate()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q_input = torch.randn(shape.batch, shape.heads, 128, device=device, dtype=shape.torch_dtype, generator=gen)
    weight = torch.randn(shape.batch, shape.heads, device=device, dtype=shape.torch_dtype, generator=gen)
    freqs_cis = make_freqs_cis(shape.max_seq_len, 64, device)
    positions = torch.randint(0, shape.max_seq_len, (shape.batch,), device=device, dtype=torch.int32, generator=gen)
    return q_input, weight, 0.03125, freqs_cis, positions


def _rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    tail = out[..., -64:].float().reshape(*out.shape[:-1], 32, 2)
    freqs = torch.view_as_real(freqs_cis[positions]).view(x.shape[0], 1, 32, 2)
    real = tail[..., 0] * freqs[..., 0] - tail[..., 1] * freqs[..., 1]
    imag = tail[..., 0] * freqs[..., 1] + tail[..., 1] * freqs[..., 0]
    out[..., -64:] = torch.stack((real, imag), dim=-1).reshape(*out.shape[:-1], 64).to(out.dtype)
    return out


def _sglang_hadamard_128_order(x: torch.Tensor) -> torch.Tensor:
    data = x.float().view(*x.shape[:-1], 32, 4)
    a0, a1, a2, a3 = data.unbind(dim=-1)
    local = torch.stack((a0 + a1, a0 - a1, a2 + a3, a2 - a3), dim=-1)
    b0, b1, b2, b3 = local.unbind(dim=-1)
    data = torch.stack((b0 + b2, b1 + b3, b0 - b2, b1 - b3), dim=-1)
    for mask in (1, 2, 4, 8, 16):
        idx = torch.arange(32, device=x.device)
        other = data.index_select(-2, idx ^ mask)
        sign = torch.where((idx & mask) == 0, 1.0, -1.0).view(*([1] * (data.ndim - 2)), 32, 1)
        data = other + sign * data
    return (data * (1.0 / (128.0 ** 0.5))).reshape_as(x)


def eager_forward(q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor):
    hadamard = _sglang_hadamard_128_order(_rope_tail(q_input, freqs_cis, positions))
    scale = torch.clamp(hadamard.abs().amax(dim=-1, keepdim=True), min=1e-4) / FP8_E4M3_MAX
    q_fp8 = (hadamard / scale).to(torch.float8_e4m3fn)
    weights_out = weight.float().unsqueeze(-1) * float(weight_scale) * scale
    return q_fp8, weights_out


def compiler_forward(q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor):
    global _COMPILED_EAGER
    if _COMPILED_EAGER is None:
        _COMPILED_EAGER = torch.compile(eager_forward, mode="max-autotune", fullgraph=False)
    return _COMPILED_EAGER(q_input, weight, weight_scale, freqs_cis, positions)


def sglang_forward(q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor):
    _prepend_python_bin_to_path()
    _patch_flashinfer_ninja()
    fused = _load_sglang_dsv4_elementwise().fused_q_indexer_rope_hadamard_quant
    return fused(q_input.contiguous(), weight.contiguous(), float(weight_scale), freqs_cis, positions)


def candidate_forward(q_input: torch.Tensor, weight: torch.Tensor, weight_scale: float, freqs_cis: torch.Tensor, positions: torch.Tensor):
    return custom_ops.indexer(q_input, weight, weight_scale, freqs_cis, positions)
