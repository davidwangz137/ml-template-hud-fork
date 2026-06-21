#!/usr/bin/env python
"""Local runner for a single HUD task -- mirrors modal_runner.run_agent, minus Modal.

Runs the task's scenario + graders *in-process on this machine* (your local GPU),
driving an LLM agent against a locally staged workspace. Nothing touches Modal.

Prereqs:
  - `uv sync` so the repo .venv has hud, torch, mteb, ...
  - HUD authenticated (already done): ~/.hud/.env with HUD_API_KEY
  - For --direct-openai: an OpenAI API key file (default below).

Usage:
  # Gateway agent (LLM routed via HUD gateway; identical path to modal_runner):
  uv run python local_runner.py --task perf_debug_sync --model claude-sonnet-4-5 --max-steps 60

  # Direct OpenAI with YOUR key (no gateway hop for LLM calls; uses OpenAIChatAgent):
  uv run python local_runner.py --task perf_debug_sync --direct-openai --model gpt-4o --max-steps 60

Differences from modal_runner (all safe locally):
  - SRC_DIR defaults to THIS repo root (not /mcp_server); WORKSPACE defaults to
    /tmp/hud_workspace. _setup_workspace() rmtrees WORKSPACE, so keep it out of the repo.
  - No /opt/venv staged link (guarded by isdir in env.py); the agent's bash uses the
    active interpreter -- invoke via `uv run` so it's the repo .venv.
  - run_train.sh defaults NGPU=8; the agent should set NGPU=1 for a single local GPU.
  - The tps>=8000 throughput floor is H100-calibrated and will NOT clear on a laptop
    GPU even when fixed. The code-fix / checkpoint / retrieval-eval graders all run.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent
DEFAULT_OPENAI_KEY_FILE = pathlib.Path.home() / "code_hackathon" / "keys" / "openai_api.txt"


def load_task(slug: str):
    """Import tasks/<slug>/task.py and return its `task` object (matches modal_runner)."""
    task_dir = REPO / "tasks" / slug
    if not task_dir.is_dir():
        raise SystemExit(f"Unknown task slug: {slug} (no {task_dir})")
    sys.path.insert(0, str(task_dir))
    if "task" in sys.modules:  # ensure re-import between runs
        del sys.modules["task"]
    task = importlib.import_module("task").task
    sys.path.pop(0)
    return task


def localize_paths(task) -> None:
    """Remap container-absolute paths in the task setup_command to this repo.

    Task definitions hardcode /mcp_server/... (the container SRC_DIR). Locally
    SRC_DIR is this repo, and setup_fixtures.py lives at tasks/utils/ (there is
    no setup/ dir, and /mcp_server isn't writable without root). setup_fixtures.py
    itself is SRC_DIR-aware, so only the *invocation path* needs rewriting.
    """
    cmd = task.args.get("setup_command") or ""
    cmd = cmd.replace(
        "/mcp_server/setup/setup_fixtures.py",
        str(REPO / "tasks" / "utils" / "setup_fixtures.py"),
    )
    task.args["setup_command"] = cmd


def build_agent(model: str, direct_openai: bool, key_file: pathlib.Path,
                system_prompt: str, max_steps: int):
    """Gateway agent (default, mirrors modal_runner) or direct-OpenAI agent.

    Both system_prompt and max_steps are OpenAIChatConfig fields, so they go
    through config construction. Auth nuance: settings.api_key (HUD gateway key)
    takes precedence, so for direct OpenAI we pass an explicit model_client --
    setting OPENAI_API_KEY alone is insufficient while HUD_API_KEY is present.
    """
    if direct_openai:
        if not key_file.is_file():
            raise SystemExit(f"--direct-openai set but key file missing: {key_file}")
        api_key = key_file.read_text().strip()
        os.environ["OPENAI_API_KEY"] = api_key  # for downstream settings reads
        from openai import AsyncOpenAI

        # Build via the AgentType registry -- same path create_agent() uses.
        # Use native OpenAIAgent, not openai_compatible: openai_compatible only
        # exposes read/grep/glob/list, while native OpenAIAgent exposes the shell
        # tool needed to run training and edit/fix code.
        from hud.types import AgentType

        at = AgentType("openai")  # -> OpenAIAgent (native tools incl. shell)
        cfg = at.config_cls(
            model=model,
            model_client=AsyncOpenAI(api_key=api_key),
            system_prompt=system_prompt,
            max_steps=max_steps,
            auto_respond=True,
        )
        return at.cls(cfg)

    from hud.agents import create_agent

    # create_agent() routes ALL LLM calls through the HUD gateway (uses HUD_API_KEY).
    return create_agent(model, system_prompt=system_prompt, max_steps=max_steps, auto_respond=True)


def _shorten(value: object, limit: int = 160) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_step(step) -> str:
    data = step.model_dump(mode="json", exclude_none=True)
    parts = [f"source={data.get('source')}"]
    if task_call := data.get("task_call"):
        task = task_call.get("name")
        phase = task_call.get("phase")
        parts.append(f"task={task}/{phase}")
    if call := data.get("call"):
        parts.append(f"tool={call.get('name')}")
    if tool_calls := data.get("tool_calls"):
        names = ",".join(call.get("name", "?") for call in tool_calls)
        parts.append(f"tool_calls={names}")
    if data.get("done") is True:
        parts.append("done=true")
    if error := data.get("error"):
        parts.append(f"error={_shorten(error)}")
    if content := data.get("content"):
        parts.append(f"content={_shorten(content)}")
    return " ".join(parts)


def install_step_stream(path: pathlib.Path | None):
    """Mirror each live Run.record() step to stdout and optional JSONL."""
    from hud.eval.run import Run

    original_record = Run.record
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def streaming_record(self, step):
        original_record(self, step)
        idx = len(self.trace.steps)
        print(f"Step[{idx}] {_summarize_step(step)}", flush=True)
        if path is not None:
            with path.open("a") as f:
                f.write(json.dumps(step.model_dump(mode="json", exclude_none=True)) + "\n")

    Run.record = streaming_record
    return original_record


async def run(args: argparse.Namespace) -> None:
    # env.py reads these at import time; set before `import env`. LocalRuntime
    # serves env.py in a CHILD process, which inherits these via os.environ.
    os.environ.setdefault("SRC_DIR", str(REPO))
    os.environ.setdefault("WORKSPACE", args.workspace)
    os.environ.setdefault("HUD_WORKSPACE_ROOT", args.workspace)
    os.environ.setdefault("MCP_TESTING_MODE", "1")
    os.environ.setdefault("HUD_TELEMETRY_ENABLED", "false")
    local_venv = REPO / ".venv"
    if local_venv.is_dir():
        os.environ.setdefault("HUD_LOCAL_VENV", str(local_venv))

    import env
    from env import AGENT_CONFIG
    from hud.eval import rollout, LocalRuntime

    # AGENT_CONFIG's prompt hardcodes the container path /home/ubuntu/workspace.
    # In hud v6 LocalRuntime, the host WORKSPACE is mounted inside bwrap at
    # /workspace; point the agent at the guest path it can actually use.
    sys_prompt = AGENT_CONFIG["system_prompt"].replace("/home/ubuntu/workspace", "/workspace")

    task = load_task(args.task)
    localize_paths(task)
    print(f"    setup_command: {task.args.get('setup_command')}")
    try:
        task.metadata["trace_name"] = args.task  # cosmetic; absent on some hud builds
    except (AttributeError, TypeError):
        pass
    task.agent_config = AGENT_CONFIG

    mode = "direct-openai" if args.direct_openai else "gateway"
    print(f"=== {args.task} ({args.model}, {mode}) | workspace={args.workspace} "
          f"| max_steps={args.max_steps} | timeout={args.timeout}s ===")
    agent = build_agent(args.model, args.direct_openai, args.openai_key_file,
                        sys_prompt, args.max_steps)

    # 0.6.6 API: rollout(task, agent, runtime=LocalRuntime(env.py)) -> Run.
    # LocalRuntime serves env.py in a child process on a loopback port; rollout
    # drives the agent (here) against that channel and grades on exit (run.reward).
    stream_path = args.stream_steps
    original_record = install_step_stream(stream_path)
    try:
        run_obj = await rollout(
            task,
            agent,
            runtime=LocalRuntime(str(REPO / "env.py"), env="ml-template-1", ready_timeout=240.0),
            rollout_timeout=args.timeout,
        )
    finally:
        from hud.eval.run import Run
        Run.record = original_record
    if stream_path is not None:
        print(f"Step JSONL: {stream_path}")
    print(f"Reward: {run_obj.reward}")
    pathlib.Path("/tmp/local_run_trace.json").write_text(run_obj.trace.model_dump_json(indent=2))
    print("Trace JSON: /tmp/local_run_trace.json")
    print(f"Trace status: {run_obj.trace.status} extra={run_obj.trace.extra}")
    for idx, step in enumerate(run_obj.trace.steps):
        task_name = step.task_call.name if step.task_call else None
        msg_count = len(step.messages or [])
        print(
            f"Trace[{idx}] source={step.source} task={task_name} "
            f"messages={msg_count} error={step.error!r}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="Task slug under tasks/ (e.g. perf_debug_sync)")
    p.add_argument("--model", default="gpt-4o", help="Model name (direct) or gateway model id")
    p.add_argument("--max-steps", type=int, default=60, help="Agent action budget")
    p.add_argument("--direct-openai", action="store_true", help="Use YOUR OpenAI key (no gateway)")
    p.add_argument("--openai-key-file", type=pathlib.Path, default=DEFAULT_OPENAI_KEY_FILE)
    p.add_argument("--workspace", default="/tmp/hud_workspace", help="Staged workspace dir (will be wiped)")
    p.add_argument("--timeout", type=int, default=1800, help="rollout_timeout seconds (wall clock)")
    p.add_argument(
        "--stream-steps",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/local_run_steps.jsonl"),
        help="Write each live trace step as JSONL while the rollout runs.",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
