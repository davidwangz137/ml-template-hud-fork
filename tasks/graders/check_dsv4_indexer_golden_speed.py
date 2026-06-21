"""Compare a candidate DSV4 indexer solution against the hidden CUDA golden.

This grader runs the public speed/parity grader twice in isolated subprocesses:

1. against the candidate workspace passed by HUD;
2. against the repository's reference_solution_files, which are not staged into
   the agent workspace.

The agent can see neither the reference CUDA source nor this grader during its
rollout. The comparison is only performed after the answer is submitted.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SPEED_RE = re.compile(r"geomean_vs_sglang=([0-9.]+)")


def _run_speed_grader(src_dir: Path, target: Path, label: str) -> tuple[float, str]:
    with tempfile.TemporaryDirectory(prefix=f"dsv4_{label}_ext_") as ext_dir:
        env = os.environ.copy()
        env["TORCH_EXTENSIONS_DIR"] = ext_dir
        python_bin = Path(sys.executable).resolve().parent
        env["PATH"] = f"{python_bin}:{env.get('PATH', '')}"
        proc = subprocess.run(
            [sys.executable, str(src_dir / "tasks" / "graders" / "check_dsv4_indexer_speed.py"), str(target)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(src_dir),
            timeout=900,
            check=False,
        )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"{label} speed grader failed with exit {proc.returncode}\n{output}")
    match = SPEED_RE.search(output)
    if match is None:
        raise RuntimeError(f"{label} speed grader did not report geomean_vs_sglang\n{output}")
    return float(match.group(1)), output


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_dsv4_indexer_golden_speed.py <workspace> <src_dir>")
        return 1
    workspace = Path(argv[0]).resolve()
    src_dir = Path(argv[1]).resolve()
    golden_root = src_dir / "tasks" / "sglang_dsv4_indexer" / "reference_solution_files"
    if not (workspace / "dsv4_indexer" / "indexer.py").exists():
        print(f"candidate dsv4_indexer package not found under {workspace}")
        return 1
    if not (golden_root / "dsv4_indexer" / "indexer.py").exists():
        print(f"golden dsv4_indexer package not found under {golden_root}")
        return 1

    try:
        candidate_speed, candidate_output = _run_speed_grader(src_dir, workspace, "candidate")
        golden_speed, golden_output = _run_speed_grader(src_dir, golden_root, "golden")
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc))
        return 1

    ratio = candidate_speed / golden_speed if golden_speed > 0 else 0.0
    # 1.0 means matching or beating the hidden golden. Partial credit is linear.
    score = max(0.0, min(1.0, ratio))
    print("=== candidate speed grader ===")
    print(candidate_output.rstrip())
    print("=== hidden golden speed grader ===")
    print(golden_output.rstrip())
    print(f"candidate_geomean_vs_sglang={candidate_speed:.4f}")
    print(f"golden_geomean_vs_sglang={golden_speed:.4f}")
    print(f"candidate_vs_hidden_golden={ratio:.4f}")
    print(f"SCORE: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
