"""HUD environment for ML training tasks built on torchtitan experiments."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hud import Environment
from hud.environment import workspace as _ws_mod
from hud.environment.workspace import DEFAULT_SYSTEM_MOUNTS, Mount
from hud.graders import BashGrader, SubScore, combine

logger = logging.getLogger(__name__)
MCP_TESTING_MODE = os.environ.get("MCP_TESTING_MODE") in ["1", "true"]


def bash(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    logger.info("bash: %s", cmd)
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning(
            "bash exited %d\nstdout: %s\nstderr: %s",
            result.returncode,
            result.stdout[-3000:] if result.stdout else "(empty)",
            result.stderr[-3000:] if result.stderr else "(empty)",
        )
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result



# Locations are overridable so the same env serves both the Modal/Docker image
# (the defaults) and ad-hoc local runs.
SRC_DIR = os.environ.get("HUD_SRC_DIR", "/mcp_server")
WORKSPACE = os.environ.get("HUD_WORKSPACE", "/workspace")
STAGED_VENV = os.environ.get("HUD_VENV", "/opt/venv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

env = Environment(name="ml-template-1")


def _patch_workspace_for_local_cuda() -> None:
    """Bind WSL/NVIDIA device nodes into local bwrap shells when present."""
    if not os.environ.get("HUD_LOCAL_VENV"):
        return

    from hud.environment.workspace import Workspace

    if getattr(Workspace, "_ml_template_cuda_patch", False):
        return

    original_bwrap_argv = Workspace.bwrap_argv

    def bwrap_argv_with_cuda(self, command, *, cwd=None, env=None):
        argv = original_bwrap_argv(self, command, cwd=cwd, env=env)
        devices = (
            "/dev/dxg",          # WSL GPU device
            "/dev/nvidiactl",
            "/dev/nvidia0",
            "/dev/nvidia-uvm",
            "/dev/dri",
        )
        extra = []
        for device in devices:
            if Path(device).exists():
                extra.extend(["--dev-bind-try", device, device])
        if extra:
            argv[argv.index("--clearenv"):argv.index("--clearenv")] = extra
        return argv

    Workspace.bwrap_argv = bwrap_argv_with_cuda
    Workspace._ml_template_cuda_patch = True


_patch_workspace_for_local_cuda()


def _local_workspace_kwargs() -> dict[str, Any]:
    """Expose the repo venv read-only inside the local bwrap workspace."""
    venv = os.environ.get("HUD_LOCAL_VENV")
    if not venv:
        return {}

    venv_path = Path(venv).resolve()
    mounts = [Mount("ro", src=str(venv_path), dst=str(venv_path))]

    # uv-created venvs often symlink .venv/bin/python through the uv-managed
    # interpreter directory outside the repo. Mount the containing uv/python
    # directory read-only so both the stable symlink path and the versioned
    # interpreter target resolve inside bwrap.
    python_bin = venv_path / "bin" / "python"
    if python_bin.exists():
        target = python_bin.resolve()
        if not str(target).startswith(str(venv_path)):
            mounts.append(Mount("ro", src=str(target.parents[2]), dst=str(target.parents[2])))

    path = os.environ.get("PATH", "/usr/bin:/bin")
    env_overrides = {
        "PATH": f"{venv_path / 'bin'}:{path}",
        "VIRTUAL_ENV": str(venv_path),
        "PYTHONPATH": WORKSPACE,
    }

    # WSL exposes the NVIDIA userspace driver here. Harmless on non-WSL hosts
    # where the path does not exist; PATH also lets nvidia-smi resolve locally.
    if Path("/usr/lib/wsl/lib").is_dir():
        env_overrides["PATH"] = f"{venv_path / 'bin'}:/usr/lib/wsl/lib:{path}"
        ld_library_path = os.environ.get("LD_LIBRARY_PATH")
        env_overrides["LD_LIBRARY_PATH"] = (
            f"/usr/lib/wsl/lib:{ld_library_path}" if ld_library_path else "/usr/lib/wsl/lib"
        )

    return {"mounts": mounts, "env": env_overrides}


_workspace_kwargs = _local_workspace_kwargs()
if _workspace_kwargs:
    # Register before legacy tool setup. The legacy SSH adapter skips creating
    # its default workspace when an ssh capability already exists.
    env.workspace(WORKSPACE, **_workspace_kwargs)

AGENT_CONFIG = {
    "system_prompt": (
        "You are an expert ML engineer working with torchtitan.\n\n"
        "Environment:\n"
        "  Workspace: /workspace (your shell starts here)\n"
        "  Framework source: /workspace/torchtitan  (editable)\n"
        "  Pre-staged data and assets: /workspace/data/, /workspace/assets/\n\n"
        "Running experiments:\n"
        "  Standard models train via run_train.sh (wraps torchrun -m torchtitan.train):\n"
        "    NGPU=1 MODULE=<module> CONFIG=<config> ./run_train.sh [extra CLI flags]\n"
        "  Experiments under torchtitan/experiments/ may define their own entrypoints.\n"
        "  Check the experiment code for usage.\n\n"
        "Constraints:\n"
        "  - Bash sessions time out after 120s with no output. For training runs:\n"
        "      nohup cmd > log.log 2>&1 & echo PID:$! -- then poll with: tail -20 log.log\n"
        "  - Delete intermediate checkpoints to save disk, but always keep your final checkpoint.\n"
    ),
}


# ===========================================================================
# Workspace capability (bwrap-isolated SSH shell with GPU access)
# ===========================================================================

# Use a device bind for GPU access inside the isolated workspace.
if "devbind" not in _ws_mod._MOUNT_FLAGS:
    _ws_mod._MOUNT_FLAGS["devbind"] = ("--dev-bind", "--dev-bind-try", True)


def _gpu_system_mounts() -> tuple[Mount, ...]:
    """Default system mounts, but with the minimal ``--dev`` swapped for a real
    ``--dev-bind /dev /dev`` so the NVIDIA device nodes are usable."""
    return tuple(
        Mount("devbind", src="/dev", dst="/dev") if m.kind == "dev" else m
        for m in DEFAULT_SYSTEM_MOUNTS
    )


def _workspace_mounts() -> list[Mount]:
    """Read-only mounts visible inside the sandbox (the Python venv)."""
    mounts: list[Mount] = []
    if os.path.isdir(STAGED_VENV):
        mounts.append(Mount("ro", src=STAGED_VENV, dst=STAGED_VENV))
    return mounts


def _gpu_present() -> bool:
    """Whether this container appears to have NVIDIA device nodes attached."""
    return any(Path(path).exists() for path in ("/dev/nvidiactl", "/dev/nvidia0"))


# One workspace per env: an SSH shell the agent drives. ``guest_path=WORKSPACE``
# mounts the root at its real path inside the sandbox so the agent's paths and
# the host-side grader paths coincide. ``network=True`` keeps the host net
# namespace so torchrun's localhost rendezvous works.
workspace = env.workspace(
    WORKSPACE,
    network=True,
    guest_path=WORKSPACE,
    system_mounts=_gpu_system_mounts(),
    mounts=_workspace_mounts(),
    env={
        "PATH": f"{STAGED_VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": WORKSPACE,
        "HOME": WORKSPACE,
        "HF_HOME": f"{WORKSPACE}/.cache/huggingface",
    },
)


@env.initialize
async def _probe_bwrap() -> None:
    """Verify bwrap isolation can run before exposing the workspace.

    ``Workspace`` uses bwrap whenever the binary is on PATH, but some container
    runtimes ship it while forbidding the user namespace / device bind it needs
    (every SSH command would then fail). Probe the real workspace configuration
    once; only allow an unisolated fallback for local/debug startup where there
    are no GPU device nodes or where the override is explicit.
    """
    import asyncio

    if workspace._bwrap is None:
        return
    os.makedirs(WORKSPACE, exist_ok=True)
    argv = workspace.bwrap_argv(
        ["bash", "-lc", "echo probe > /dev/null && nvidia-smi -L >/dev/null 2>&1 || true"],
        cwd=WORKSPACE,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
        if proc.returncode != 0:
            raise RuntimeError((stderr or b"").decode(errors="replace")[-500:] or "non-zero exit")
    except Exception as exc:  # noqa: BLE001 - any failure means bwrap is unusable
        message = (
            "bwrap is present but unusable here "
            f"({exc}); refusing to serve an unisolated workspace in this configuration."
        )
        allow_unisolated = os.environ.get("HUD_ALLOW_UNISOLATED_WORKSPACE") == "1"
        if allow_unisolated or not _gpu_present():
            logger.warning(
                "%s The workspace will run WITHOUT isolation. This is allowed "
                "only because %s.",
                message,
                "HUD_ALLOW_UNISOLATED_WORKSPACE=1 is set"
                if allow_unisolated
                else "no NVIDIA device nodes are present, so this is deploy introspection/local no-GPU startup",
            )
            workspace._bwrap = None
            return
        raise RuntimeError(
            f"{message} Set HUD_ALLOW_UNISOLATED_WORKSPACE=1 only for local debugging."
        ) from exc
    else:
        logger.info("bwrap isolation active for the workspace shell.")


# ===========================================================================
# Grading (host-side, outside the sandbox)
# ===========================================================================


async def _score_command_grader(g: dict[str, Any], weight: float) -> SubScore:
    """Run a grader command that prints SCORE: <0..1> for partial credit."""
    command = g["command"]
    timeout_seconds = g.get("timeout", 10)
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else 1
    except TimeoutError:
        return SubScore(
            name=g.get("name"),
            weight=weight,
            value=0.0,
            metadata={"exit_code": None, "stdout": "", "stderr": "", "timed_out": True, "timeout": timeout_seconds},
        )

    score = 0.0
    if exit_code == 0:
        for line in stdout.splitlines()[::-1]:
            if line.startswith("SCORE:"):
                try:
                    score = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    score = 0.0
                break
    return SubScore(
        name=g.get("name"),
        weight=weight,
        value=score,
        metadata={"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
    )


async def _grade(graders: list[dict[str, Any]]):
    """Build an ``EvaluationResult`` from a list of grader dicts.

    Each grader is either:
      - script-based: {name, script, _script_stem?, args?, weight?, timeout?}
        The script is written to /tmp/<stem>.py and the command auto-built.
      - command-based: {name, command, weight?, timeout?}  (runs as-is)
      - partial-score command: add score_stdout=True and print SCORE: <0..1>.

    Graders run host-side via ``BashGrader`` with ``cwd=WORKSPACE`` so they see
    the agent's edits; ``combine`` normalizes the positive weights to sum to 1.
    """
    # Kill any leftover agent GPU processes so graders have full GPU access.
    os.system("pkill -9 -f torchrun 2>/dev/null; pkill -9 -f torchtitan.train 2>/dev/null; sleep 1")

    subscores = []
    for g in graders:
        if "script" in g:
            stem = g.get("_script_stem", g["name"])
            Path(f"/tmp/{stem}.py").write_text(g["script"])
            command = g.get("command") or f"python /tmp/{stem}.py {g.get('args', '')}"
        else:
            command = g["command"]
        if g.get("score_stdout"):
            subscores.append(_score_command_grader(g, g.get("weight", 1)))
        else:
            subscores.append(
                BashGrader.grade(
                    weight=g.get("weight", 1),
                    name=g.get("name"),
                    command=command,
                    cwd=WORKSPACE,
                    timeout_seconds=g.get("timeout", 10),
                )
            )
    return await combine(*subscores)


# ===========================================================================
# Workspace seeding
# ===========================================================================


def _clean_workspace() -> None:
    """Empty the workspace while preserving HUD runtime state."""
    workspace_path = Path(WORKSPACE).expanduser().resolve()
    src_path = Path(SRC_DIR).expanduser().resolve()
    if (
        workspace_path == Path("/").resolve()
        or workspace_path == Path.home().resolve()
        or src_path == workspace_path
        or src_path.is_relative_to(workspace_path)
    ):
        raise RuntimeError(f"Refusing to clean unsafe WORKSPACE={workspace_path}")
    workspace_path.mkdir(parents=True, exist_ok=True)
    for child in workspace_path.iterdir():
        if child.name == ".hud":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _setup_workspace(setup_command: str | None = None) -> None:
    """Seed the workspace and optionally run a command to stage assets.

    Copies torchtitan source, tests, and the standard launcher script, and
    symlinks the relocatable venv. Task-specific assets are downloaded by the
    setup_command. Runs host-side (full access), unlike the agent's shell.
    """
    _clean_workspace()

    bash(f"cp -r {SRC_DIR}/torchtitan {WORKSPACE}/torchtitan")
    bash(f"cp -r {SRC_DIR}/tests {WORKSPACE}/tests")
    if os.path.exists(f"{SRC_DIR}/run_train.sh"):
        bash(f"cp {SRC_DIR}/run_train.sh {WORKSPACE}/run_train.sh")
        bash(f"chmod +x {WORKSPACE}/run_train.sh")

    if os.path.isdir(STAGED_VENV) and not os.path.lexists(f"{WORKSPACE}/.venv"):
        bash(f"ln -s {STAGED_VENV} {WORKSPACE}/.venv")

    if setup_command:
        bash(setup_command)

    os.chdir(WORKSPACE)


def _apply_patches(patches: list[str] | None = None) -> None:
    """Apply inline patch strings to the workspace."""
    if not patches:
        return

    for i, patch_content in enumerate(patches):
        patch_path = Path(WORKSPACE) / f".patch_{i}"
        patch_path.write_text(patch_content)
        bash(f"cd {WORKSPACE} && patch --no-backup -p1 < {patch_path} && rm {patch_path}")


def _write_tmp_json(name: str, payload: dict[str, Any]) -> None:
    Path(f"/tmp/{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


# ===========================================================================
# Task templates
# ===========================================================================


@env.template(id="train_to_target")
async def train_to_target(
    prompt: str,
    graders: list[dict[str, Any]],
    setup_command: str | None = None,
):
    """Train forward from a clean or staged workspace."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.template(id="repair_degraded_recipe")
async def repair_degraded_recipe(
    prompt: str,
    graders: list[dict[str, Any]],
    patches: list[str],
    setup_command: str | None = None,
):
    """Stage a degraded recipe via patches, then let the agent repair it."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    yield prompt
    yield await _grade(graders)


@env.template(id="audit_training_data")
async def audit_training_data(
    prompt: str,
    graders: list[dict[str, Any]],
    contamination: str = "label_noise",
    train_file: str = "data/scifact.jsonl",
    val_file: str = "data/val.jsonl",
    noise_rate: float = 0.3,
    leak_rate: float = 0.2,
    setup_command: str | None = None,
):
    """Inject training-data contamination, then let the agent audit and clean it."""
    _setup_workspace(setup_command)
    bash(
        f"python -m tasks.mutations data {contamination} {WORKSPACE}"
        f" --train-file {train_file} --val-file {val_file}"
        f" --noise-rate {noise_rate} --leak-rate {leak_rate}"
    )
    yield prompt
    yield await _grade(graders)


@env.template(id="audit_evaluation_signal")
async def audit_evaluation_signal(
    prompt: str,
    graders: list[dict[str, Any]],
    eval_mutation: str = "eval_leakage",
    eval_file: str = "data/val.jsonl",
    train_file: str = "data/scifact.jsonl",
    leak_rate: float = 0.25,
    setup_command: str | None = None,
):
    """Corrupt the visible evaluation signal and require the agent to audit it."""
    _setup_workspace(setup_command)
    bash(
        f"python -m tasks.mutations eval {eval_mutation} {WORKSPACE}"
        f" --eval-file {eval_file} --train-file {train_file}"
        f" --leak-rate {leak_rate}"
    )
    yield prompt
    yield await _grade(graders)


@env.template(id="compose_multi_stage_pipeline")
async def compose_multi_stage_pipeline(
    prompt: str,
    graders: list[dict[str, Any]],
    expected_stages: list[str] | None = None,
    setup_command: str | None = None,
):
    """Encourage staged training pipelines with intermediate artifacts."""
    _setup_workspace(setup_command)
    if expected_stages is not None:
        _write_tmp_json("pipeline_spec", {"expected_stages": expected_stages})
    yield prompt
    yield await _grade(graders)


@env.template(id="certify_reliability")
async def certify_reliability(
    prompt: str,
    graders: list[dict[str, Any]],
    reliability_matrix: list[dict[str, Any]],
    patches: list[str] | None = None,
    setup_command: str | None = None,
):
    """Require evidence that a recipe is stable across a small run matrix."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    _write_tmp_json("reliability_spec", {"matrix": reliability_matrix})
    yield prompt
    yield await _grade(graders)


@env.template(id="optimize_under_constraints")
async def optimize_under_constraints(
    prompt: str,
    graders: list[dict[str, Any]],
    constraints: dict[str, Any],
    setup_command: str | None = None,
):
    """Expose explicit resource or experiment-budget constraints to the agent."""
    _setup_workspace(setup_command)
    _write_tmp_json("constraint_spec", constraints)
    yield prompt
    yield await _grade(graders)


@env.template(id="adapt_without_forgetting")
async def adapt_without_forgetting(
    prompt: str,
    graders: list[dict[str, Any]],
    base_checkpoint: str,
    adapt_train_files: list[str],
    retain_eval_files: list[str],
    forbidden_train_files: list[str] | None = None,
    setup_command: str | None = None,
):
    """Stage a base checkpoint and require adaptation to new data without forgetting."""
    _setup_workspace(setup_command)
    for rel_path in forbidden_train_files or []:
        forbidden_path = Path(WORKSPACE) / rel_path
        if forbidden_path.exists():
            forbidden_path.unlink()
    yield prompt
    yield await _grade(graders)


@env.template(id="targeted_failure_recovery")
async def targeted_failure_recovery(
    prompt: str,
    graders: list[dict[str, Any]],
    failure_manifest: dict[str, Any] | None = None,
    setup_command: str | None = None,
):
    """Stage failing artifacts or subsets and require targeted recovery."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.template(id="restore_reference_parity")
async def restore_reference_parity(
    prompt: str,
    graders: list[dict[str, Any]],
    reference_spec: dict[str, Any],
    patches: list[str] | None = None,
    setup_command: str | None = None,
):
    """Stage a reference artifact or spec and require parity restoration."""
    _setup_workspace(setup_command)
    _apply_patches(patches)
    _write_tmp_json("reference_spec", reference_spec)
    yield prompt
    yield await _grade(graders)


SCENARIOS = {
    name: value
    for name, value in globals().items()
    if not name.startswith("_") and name in env.templates
}
