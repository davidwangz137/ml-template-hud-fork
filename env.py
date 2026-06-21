"""ML training environment -- torchtitan experiments."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hud import Environment
from hud.environment import Mount

try:
    from hud.graders import BashGrader, SubScore, combine

    class Grade:
        gather = staticmethod(combine)
except ModuleNotFoundError:
    import asyncio

    from hud.tools.types import EvaluationResult, SubScore

    class BashGrader:
        @classmethod
        async def grade(
            cls,
            *,
            name: str,
            weight: float,
            command: str,
            timeout_seconds: int,
        ) -> SubScore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash",
                    "-lc",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
                metadata = {
                    "exit_code": proc.returncode,
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                }
                value = 1.0 if proc.returncode == 0 else 0.0
            except TimeoutError:
                proc.kill()
                await proc.wait()
                metadata = {"exit_code": None, "timed_out": True}
                value = 0.0
            return SubScore(name=name, weight=weight, value=value, metadata=metadata)

    class Grade:
        @staticmethod
        async def gather(*items):
            subscores = list(await asyncio.gather(*items))
            reward = sum(item.value * item.weight for item in subscores)
            return EvaluationResult(reward=reward, done=True, subscores=subscores)


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

SRC_DIR = os.environ.get("SRC_DIR", "/mcp_server")
WORKSPACE = os.environ.get("WORKSPACE", "/home/ubuntu/workspace")

import sys
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

env = Environment("ml-template-1")


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
        "  Workspace: /home/ubuntu/workspace (your shell starts here)\n"
        "  Framework source: /home/ubuntu/workspace/torchtitan  (editable)\n"
        "  No network access.\n"
        "  Pre-staged data and assets: /home/ubuntu/workspace/data/, /home/ubuntu/workspace/assets/\n\n"
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

_tools_initialized = False


def init_tools():
    """Initialize coding tools with workspace sandboxing."""
    global _tools_initialized
    if _tools_initialized:
        return

    # Lock down /mcp_server so the agent can't read source, graders, or patches.
    # Done at runtime (not just Dockerfile) because Modal adds files after build.
    if os.getuid() == 0:
        if os.path.isdir("/mcp_server"):
            os.system(
                "find /mcp_server -maxdepth 0 -exec chmod 700 {} + && "
                "find /mcp_server -mindepth 1 -maxdepth 1 "
                "! -name assets ! -name data "
                "-exec chmod -R 700 {} +"
            )
        os.system("chmod -R 700 /tmp/.grader_* 2>/dev/null || true")
        os.system("mount -o remount,hidepid=2 /proc 2>/dev/null || true")
    _tools_initialized = True

    import asyncio as _aio

    from hud.tools.coding import (
        ApplyPatchTool,
        BashTool,
        ClaudeBashSession,
        EditTool,
        GeminiEditTool,
        GeminiShellTool,
        GeminiWriteTool,
        ShellTool,
    )
    from hud.tools.coding.utils import get_demote_preexec_fn
    from hud.tools.filesystem import (
        GeminiGlobTool,
        GeminiListTool,
        GeminiReadManyTool,
        GeminiReadTool,
        GeminiSearchTool,
    )

    ws = WORKSPACE

    class _SandboxedSession(ClaudeBashSession):
        """Bash session locked to the workspace directory."""

        async def start(self):
            if self._started:
                await _aio.sleep(0)
                return
            self._process = await _aio.create_subprocess_shell(
                self.command,
                stdin=_aio.subprocess.PIPE,
                stdout=_aio.subprocess.PIPE,
                stderr=_aio.subprocess.PIPE,
                cwd=ws,
                preexec_fn=get_demote_preexec_fn(),
            )
            self._started = True
            self._timed_out = False
            await self.run(
                f'export HOME="{ws}" && '
                f'export PATH="{ws}/.venv/bin:/usr/bin:/bin" && '
                f'export PYTHONPATH="{ws}" && '
                f'_ws="{ws}" && '
                f'cd() {{ local t="${{1:-.}}"; local r=$(realpath -m "$t" 2>/dev/null || echo "$t"); '
                f'case "$r" in "$_ws"*) builtin cd "$t" ;; '
                f'*) echo "Error: cannot navigate outside workspace" >&2; return 1 ;; esac; }} && '
                f'ls() {{ for a in "$@"; do case "$a" in -*) ;; /*) '
                f'case "$a" in "$_ws"*) ;; *) echo "Error: cannot list outside workspace" >&2; return 1 ;; esac ;; esac; done; '
                f'command ls "$@"; }} && '
                f'find() {{ case "$1" in "$_ws"*|.*) command find "$@" ;; '
                f'*) echo "Error: cannot search outside workspace" >&2; return 1 ;; esac; }} && '
                f'_check_path() {{ for a in "$@"; do case "$a" in -*|"") ;; /*) '
                f'case "$a" in "$_ws"*|/dev/*|/proc/self/*) ;; *) echo "Error: cannot access outside workspace: $a" >&2; return 1 ;; esac ;; esac; done; return 0; }} && '
                f'cat() {{ _check_path "$@" && command cat "$@"; }} && '
                f'head() {{ _check_path "$@" && command head "$@"; }} && '
                f'tail() {{ _check_path "$@" && command tail "$@"; }}'
            )

    def _register_tool(tool):
        register = getattr(tool, "register", None)
        if callable(register):
            register(env)
        else:
            env.add_tool(tool)

    # Claude tools
    bash_tool = BashTool()
    bash_tool.session = _SandboxedSession()
    _register_tool(bash_tool)
    _register_tool(EditTool())

    # OpenAI tools
    _register_tool(ShellTool(cwd=ws))
    _register_tool(ApplyPatchTool())

    # Gemini tools
    _register_tool(GeminiShellTool(base_directory=ws))
    _register_tool(GeminiEditTool(base_directory=ws))
    _register_tool(GeminiWriteTool(base_directory=ws))
    _register_tool(GeminiReadTool(base_path=ws))
    _register_tool(GeminiSearchTool(base_path=ws))
    _register_tool(GeminiGlobTool(base_path=ws))
    _register_tool(GeminiListTool(base_path=ws))
    _register_tool(GeminiReadManyTool(base_path=ws))


init_tools()


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
    """Build an EvaluationResult from a list of grader dicts.

    Each grader is either:
      - script-based: {name, script, args?, weight?, timeout?}
        Written to /tmp/{name}.py, command auto-built.
      - command-based: {name, command, weight?, timeout?}
        Runs as-is.
      - partial-score command: add score_stdout=True and print SCORE: <0..1>.
    """
    # Kill any leftover agent GPU processes so graders have full GPU access.
    os.system("pkill -9 -f torchrun 2>/dev/null; pkill -9 -f torchtitan.train 2>/dev/null; sleep 1")
    for g in graders:
        if "script" in g:
            stem = g.pop("_script_stem", g["name"])
            script = g.pop("script")
            args = g.pop("args", "")
            Path(f"/tmp/{stem}.py").write_text(script)
            g.setdefault("command", f"python /tmp/{stem}.py {args}")
    total = sum(g.get("weight", 1) for g in graders)
    subscores = []
    for g in graders:
        weight = g.get("weight", 1) / total
        if g.get("score_stdout"):
            subscores.append(_score_command_grader(g, weight))
        else:
            subscores.append(
                BashGrader.grade(
                    name=g.get("name"),
                    weight=weight,
                    command=g["command"],
                    timeout_seconds=g.get("timeout", 10),
                )
            )
    return await Grade.gather(*subscores)


_STAGED_VENV = "/opt/venv"


def _setup_workspace(setup_command: str | None = None):
    """Set up workspace and optionally run a command to stage assets.

    Copies torchtitan source, tests, the standard launcher script,
    and a relocatable Python venv. Task-specific assets are downloaded
    by the setup_command.
    """
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
        # hud v6 LocalRuntime starts the SSH/SFTP workspace before scenario setup.
        # Keep the root and its live SSH credentials; clear task files only.
        if child.name == ".hud":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)

    bash(f"cp -r {SRC_DIR}/torchtitan {WORKSPACE}/torchtitan")
    bash(f"cp -r {SRC_DIR}/tests {WORKSPACE}/tests")
    if os.path.exists(f"{SRC_DIR}/run_train.sh"):
        bash(f"cp {SRC_DIR}/run_train.sh {WORKSPACE}/run_train.sh")
        bash(f"chmod +x {WORKSPACE}/run_train.sh")

    if os.path.isdir(_STAGED_VENV):
        bash(f"ln -s {_STAGED_VENV} {WORKSPACE}/.venv")

    if setup_command:
        bash(setup_command)

    if os.getuid() == 0:
        bash(f"chown -R 1000:1000 {WORKSPACE}")

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
# Scenarios
# ===========================================================================


@env.scenario(name="train_to_target")
async def train_to_target(
    prompt: str,
    graders: list[dict[str, Any]],
    setup_command: str | None = None,
):
    """Train forward from a clean or staged workspace."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="repair_degraded_recipe")
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


@env.scenario(name="audit_training_data")
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


@env.scenario(name="audit_evaluation_signal")
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


@env.scenario(name="compose_multi_stage_pipeline")
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


@env.scenario(name="certify_reliability")
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


@env.scenario(name="optimize_under_constraints")
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


@env.scenario(name="adapt_without_forgetting")
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


@env.scenario(name="targeted_failure_recovery")
async def targeted_failure_recovery(
    prompt: str,
    graders: list[dict[str, Any]],
    failure_manifest: dict[str, Any] | None = None,
    setup_command: str | None = None,
):
    """Stage failing artifacts or subsets and require targeted recovery ig."""
    _setup_workspace(setup_command)
    yield prompt
    yield await _grade(graders)


@env.scenario(name="restore_reference_parity")
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


for _scenario in env.tasks.values():
    if not hasattr(_scenario, "task"):
        _scenario.task = _scenario

SCENARIOS = dict(env.tasks)
