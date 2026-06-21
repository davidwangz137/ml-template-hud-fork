"""Speed/parity benchmark for the SGLang DSV4 indexer HUD prototype."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    candidates = [
        root / "tasks" / "sglang_dsv4_indexer" / "reference_solution_files",
        root / "reference_solution_files",
        root,
    ]
    for candidate in candidates:
        if (candidate / "dsv4_indexer" / "indexer.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        print("could not find dsv4_indexer package")
        return 1

    try:
        import torch
        from dsv4_indexer.indexer import (
            DSV4IndexerShape,
            candidate_forward,
            compiler_forward,
            eager_forward,
            make_inputs,
            sglang_forward,
        )
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}")
        return 1

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    def stats(lhs, rhs, prefix: str):
        lq, lw = lhs
        rq, rw = rhs
        return {
            f"{prefix}_q_max": (lq.float() - rq.float()).abs().max().item(),
            f"{prefix}_q_mean": (lq.float() - rq.float()).abs().mean().item(),
            f"{prefix}_w_max": (lw.float() - rw.float()).abs().max().item(),
            f"{prefix}_w_mean": (lw.float() - rw.float()).abs().mean().item(),
        }

    def time_fn(fn, q, w, ws, freqs, pos, warmup=5, iters=20):
        with torch.no_grad():
            for _ in range(warmup):
                out = fn(q, w, ws, freqs, pos)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                out = fn(q, w, ws, freqs, pos)
            end.record()
            torch.cuda.synchronize()
        if not torch.isfinite(out[1]).all():
            raise RuntimeError("non-finite output")
        return start.elapsed_time(end) / iters

    shapes = [
        DSV4IndexerShape(batch=64, heads=16, max_seq_len=8192),
        DSV4IndexerShape(batch=256, heads=16, max_seq_len=8192),
    ]
    speedups = []
    sglang_ratios = []
    for i, shape in enumerate(shapes):
        q, w, ws, freqs, pos = make_inputs(shape, 31000 + i)
        with torch.no_grad():
            eager = eager_forward(q, w, ws, freqs, pos)
            compiler = compiler_forward(q, w, ws, freqs, pos)
            sglang = sglang_forward(q, w, ws, freqs, pos)
            cand = candidate_forward(q, w, ws, freqs, pos)
        parity = {}
        parity.update(stats(sglang, eager, "sglang_vs_eager"))
        parity.update(stats(cand, sglang, "candidate_vs_sglang"))
        if parity["candidate_vs_sglang_w_max"] > 1e-6 or parity["candidate_vs_sglang_q_max"] > 0.01:
            print(f"parity too loose for {shape}: {parity}")
            return 1

        times = {"eager": [], "compiler": [], "sglang": [], "candidate": []}
        for rep in range(3):
            rq, rw, rws, rf, rp = make_inputs(shape, 41000 + i * 17 + rep)
            times["eager"].append(time_fn(eager_forward, rq, rw, rws, rf, rp))
            times["compiler"].append(time_fn(compiler_forward, rq, rw, rws, rf, rp))
            times["sglang"].append(time_fn(sglang_forward, rq, rw, rws, rf, rp))
            times["candidate"].append(time_fn(candidate_forward, rq, rw, rws, rf, rp))
        med = {k: statistics.median(v) for k, v in times.items()}
        cand_vs_eager = med["eager"] / med["candidate"]
        cand_vs_sglang = med["sglang"] / med["candidate"]
        speedups.append(cand_vs_eager)
        sglang_ratios.append(cand_vs_sglang)
        print(
            f"shape={shape} eager_ms={med['eager']:.4f} compiler_ms={med['compiler']:.4f} "
            f"sglang_ms={med['sglang']:.4f} candidate_ms={med['candidate']:.4f} "
            f"cand_vs_eager={cand_vs_eager:.3f} cand_vs_sglang={cand_vs_sglang:.3f} parity={parity}"
        )

    geomean_speedup = math.prod(max(s, 1e-6) for s in speedups) ** (1.0 / len(speedups))
    geomean_vs_sglang = math.prod(max(s, 1e-6) for s in sglang_ratios) ** (1.0 / len(sglang_ratios))
    score = max(0.0, min(1.0, geomean_vs_sglang / 1.25))
    print(f"geomean_speedup_vs_eager={geomean_speedup:.4f} geomean_vs_sglang={geomean_vs_sglang:.4f}")
    print(f"SCORE: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
