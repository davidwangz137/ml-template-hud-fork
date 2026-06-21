# Local HUD running notes

This repo can be exercised locally without Modal, but the local path differs from the Modal/H100 path.

## What runs locally

`local_runner.py` runs a HUD task through `hud.eval.rollout(...)` with `LocalRuntime`:

- parent process: `local_runner.py`
- child process: `python -m hud.environment.server env.py --env ml-template-1`
- staged workspace: `/tmp/hud_workspace`
- local GPU: the machine's CUDA GPU
- LLM path:
  - default: HUD gateway via `HUD_API_KEY`
  - `--direct-openai`: direct OpenAI via `~/code_hackathon/keys/openai_api.txt`

This is **not Docker** and **not Modal**. Modal uses `modal_runner.py` and runs in a remote container on H100.

## bwrap / sandboxing

We did **not** disable bwrap. The local HUD v6 workspace uses a bwrap-backed SSH/SFTP workspace when bwrap is available. On this machine bwrap is installed and works.

The earlier failure was not bwrap. It was:

```text
asyncssh.sftp.SFTPNoSuchFile: No such file or directory
```

Root cause: hud v6's legacy adapter roots SSH/SFTP at `HUD_WORKSPACE_ROOT` or `os.getcwd()`. It also starts that SSH workspace before the scenario setup runs. The old `_setup_workspace()` deleted the entire `WORKSPACE`, which removed the live `.hud/ssh/...` files backing the local SSH/SFTP capability.

Fixes applied:

- `local_runner.py` sets `HUD_WORKSPACE_ROOT=/tmp/hud_workspace`.
- `env.py::_setup_workspace()` now preserves `/tmp/hud_workspace/.hud/` and only clears task files.
- `local_runner.py` tells the agent to use `/workspace`, the guest path inside the bwrap workspace.

## Agent choice

Do not use the `openai_compatible` agent for this task. It only exposes
filesystem/search helpers (`read`, `grep`, `glob`, `list`) and cannot run
`run_train.sh` or patch files. A failed run with that agent will look like the
model repeatedly listing/searching files and then saying it cannot execute live
commands.

Use HUD gateway for the main smoke tests:

- `gpt-5.5` works through the HUD gateway and exposes the shell tool.
- Claude gateway models should also work if enabled for your account.

Direct OpenAI is useful only with a model/agent combination that supports HUD's
shell tool. A direct `gpt-4o` run reached OpenAI but failed because that model
rejected the `shell` tool.

Local venv exposure is now enabled for `local_runner.py`: it sets
`HUD_LOCAL_VENV=<repo>/.venv`, and `env.py` mounts that venv read-only into the
HUD bwrap shell. On this WSL GPU host, `env.py` also binds the NVIDIA device
nodes and adds `/usr/lib/wsl/lib` so `python`, `torchrun`, `nvidia-smi`, and
`torch.cuda.is_available()` work inside the agent shell.

The venv mount is read-only. Attempts to write under `<repo>/.venv` fail with
`Read-only file system`; workspace cleanup only removes `/tmp/hud_workspace`
entries, not the repo venv.

The `emb_debug_multi` task also needed a local setup fix:
`tasks/utils/setup_fixtures.py` now calls `generate_synthetic(...,
num_negatives=7)`.

## Safety

Local runs set:

```text
WORKSPACE=/tmp/hud_workspace
```

The setup code now refuses unsafe workspaces:

- `/`
- `$HOME`
- the repo root
- any parent of the repo root

So the cleanup step should only affect `/tmp/hud_workspace` task staging files, while preserving `.hud/` runtime state.

## Local task added

`tasks/perf_debug_sync` is an example profiling/performance-debug task:

- Scenario: `repair_degraded_recipe`
- Patch: injects a per-step `torch.cuda.synchronize()` into embedding training
- Graders:
  - `check_code_fix sync_stall`
  - `check_throughput`
  - checkpoint presence
  - MTEB SciFact eval
  - nDCG threshold

`check_throughput.py` parses root-level `*.log` files for torchtitan `tps:` lines and checks median tps against a floor.

Note: `tps>=8000` is H100-calibrated and will not pass on a laptop GPU even
when the fix is correct. The local run is still useful to test staging, tool
access, training, and grading mechanics.

## Run commands

Known-good local smoke test for the author-provided embedding task:

```bash
uv run python local_runner.py \
  --task emb_debug_multi \
  --model gpt-5.5 \
  --max-steps 100 \
  --timeout 3600 \
  --stream-steps /tmp/local_run_emb_debug_multi_steps.jsonl
```

What a healthy run should do:

- stage `/tmp/hud_workspace`
- expose repo `.venv` inside the agent shell
- let the agent repair code
- run `torchrun` / `torchtitan.train`
- create a checkpoint under `/tmp/hud_workspace/checkpoints/...`
- write `/tmp/hud_workspace/.emb_eval.json`
- finish with a nonzero reward

Observed reference run after the prompt patch:

```text
Reward: 0.88
Trace status: completed
Checkpoint: /tmp/hud_workspace/checkpoints/scifact_ft_repaired/checkpoint/step-1
Eval file: /tmp/hud_workspace/.emb_eval.json
```

The missing `0.12` was from exact-snippet matching in `code_fix_pooling`; the run
still trained and evaluated end-to-end.

Profiling/performance task smoke test:

```bash
uv run python local_runner.py \
  --task perf_debug_sync \
  --model gpt-5.5 \
  --max-steps 60 \
  --timeout 1800
```

Direct OpenAI example, only if the chosen model supports shell tools:

```bash
uv run python local_runner.py --task emb_debug_multi --direct-openai --model <openai-model-with-shell-tool-support>
```

## Live step streaming

`local_runner.py` mirrors each live `Run.record(...)` step to stdout and to
`/tmp/local_run_steps.jsonl` while the rollout is still running. The final full
trace is still written to `/tmp/local_run_trace.json` only after rollout
completion.

Default stream path:

```bash
/tmp/local_run_steps.jsonl
```

Override it with:

```bash
uv run python local_runner.py --task emb_debug_multi --model gpt-5.5 --stream-steps /tmp/emb_steps.jsonl
```

## Monitor command

Use this in another terminal. Change the step path if you used a custom
`--stream-steps` value.

```bash
watch -n 120 'printf "== processes ==\n"; pgrep -af "local_runner.py --task|hud.environment.server .*env.py|setup_fixtures|run_train|torchrun|torchtitan.train" || true; printf "\n== gpu ==\n"; nvidia-smi; printf "\n== recent steps ==\n"; tail -20 /tmp/local_run_emb_debug_multi_steps.jsonl 2>/dev/null || tail -20 /tmp/local_run_steps.jsonl 2>/dev/null; printf "\n== trace ==\n"; ls -lh /tmp/local_run_trace.json /tmp/hud_workspace/.emb_eval.json 2>/dev/null'
```

If the agent starts training, you should see `torchrun`/`torchtitan.train` and
GPU memory/utilization increase. If only `setup_fixtures.py` is running, the
task is still staging assets/data.

## Stopping a local run

Cancel the runner first if it is attached to your terminal. If a child HUD server
survives, stop the local task processes:

```bash
pkill -f "local_runner.py --task" || true
pkill -f "hud.environment.server .*env.py" || true
pkill -f "run_train|torchrun|torchtitan.train" || true
```

Then confirm the machine is idle:

```bash
pgrep -af "local_runner.py --task|hud.environment.server .*env.py|run_train|torchrun|torchtitan.train" || true
nvidia-smi
```

## Common failure modes

- `SSH connection closed`: usually means the local HUD workspace was deleted out
  from under the v6 SSH/SFTP capability. The current `_setup_workspace()` keeps
  `/tmp/hud_workspace/.hud/` to avoid this.
- `ModuleNotFoundError: torch` inside the agent shell: repo `.venv` was not
  mounted. `local_runner.py` should set `HUD_LOCAL_VENV` automatically when
  `.venv/` exists.
- `Tool 'shell' is not supported`: provider/model mismatch. Use the HUD gateway
  `gpt-5.5` path for this smoke test.
- Reward below `1.0` with a completed trace can still be a useful local smoke
  test. The observed `0.88` run trained, wrote a checkpoint, and evaluated; it
  lost only the exact-snippet pooling grader.
