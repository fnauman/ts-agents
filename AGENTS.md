# AGENTS.md

This repo is a CLI-first time-series analysis toolkit with:
- **Stable CLI entrypoints** (`ts-agents workflow ...`, `ts-agents tool ...`, skills, sandboxes, plus deprecated `run` / `demo` aliases)
- **Optional Gradio front ends** (`ts-agents-ui`, `ts-agents-hosted`, plus the root wrappers)
- **Canonical skills + packaged resources** for agent workflows and installed wheels

The goal of this file is to make coding agents effective and safe in this codebase.

## Quick commands (copy/paste)

### Setup
```bash
uv sync
```

### Inspect the CLI
```bash
uv run ts-agents --help
```

### Run the interactive UI
```bash
uv run ts-agents-ui
```

Source-checkout wrapper (same UI entrypoint):
```bash
uv run python main.py
```

Common UI options (see `ts-agents-ui --help`):
```bash
uv run ts-agents-ui --agent-type deep
uv run ts-agents-ui --no-agent
uv run ts-agents-ui --share
uv run ts-agents-ui --port 8080
```

### Run the hosted/manual profile
```bash
HOST=0.0.0.0 PORT=7860 uv run ts-agents-hosted
```

### Run tests
```bash
uv run python -m pytest -q
```

### Environment variables
- `TS_AGENTS_DATA_DIR` (optional): path to the dataset directory (see `ts_agents/config.py`)
- `TS_AGENTS_SANDBOX_MODE` (optional): default sandbox backend
- `OPENAI_MODEL` (optional): defaults to `gpt-5-mini` (see `ts_agents/config.py`)
- `OPENAI_API_KEY` (required for agent chat in many setups)
- Hosted profile settings are env-var driven: `HOST`, `PORT`, `GRADIO_SHARE`, `TS_AGENTS_ENABLE_AGENT`, `TS_AGENTS_AGENT_TYPE`, `TS_AGENTS_PERSIST_SESSIONS`, `TS_AGENTS_UI_TITLE`

## Repo map (where to look first)
- `ts_agents/cli/main.py` — argparse CLI for `data`, `tool`, `workflow`, `sandbox`, `agent`, `skills`, and deprecated `run` / `demo` aliases
- `main.py` — source-checkout wrapper for `ts-agents-ui`
- `app.py` — source-checkout wrapper for `ts-agents-hosted`
- `ts_agents/hosted_app.py` — hosted/manual Gradio profile configured through environment variables
- `ts_agents/ui/` — Gradio UI (tabs + chat)
- `ts_agents/agents/` — agent implementations (simple + deep)
- `ts_agents/tools/` — tool registry + wrappers (LangChain + deep agent tools)
- `ts_agents/core/` — pure analysis implementations (decomposition, forecasting, patterns, classification, spectral)
- `ts_agents/persistence/` — session persistence + caching
- `ts_agents/resources/` — packaged data, demo assets, and skill mirrors used by installed wheels
- `skills/` — canonical skill definitions
- `demo/` — source-checkout helper scripts, plots, and VHS tapes
- `docs/` — Quarto docs site source
- `tests/` — unit tests

Dependency metadata and locked versions live in `pyproject.toml` + `uv.lock`.

## Working agreement (always follow)

### GitHub workflow & issue tracking
- Use the `gh` CLI when auth is available.
- Create an issue + branch per task before coding.
- Update the issue with each incremental step.
- If `gh` auth/network is blocked, document why + next action and continue on a local branch.
- Large changes: break into smaller issues/PRs.
- PRs must explain: behavior changes, new tests, manifest/profile impacts, risks.

### Workflow for any non-trivial change
1) **Explore**: read the relevant files and summarize what you found (file paths + key functions/classes).
2) **Plan**: list *numbered* steps, including which files you will change and how you will validate each step.
3) **Implement**: execute the plan mechanically.
4) **Verify**: run the smallest relevant tests/commands and report the results.
5) **Summarize**: what changed, why, and any follow-ups.

### If something is ambiguous
Ask clarifying questions *before* implementing. If you must proceed, list assumptions explicitly.

### No “fake done”
- Do **not** use placeholder code such as “TODO implement” or “implementation goes here” and claim the task is finished.
- Do **not** claim tests passed unless you actually ran them.

### Keep LLM context small
This codebase uses plots and arrays. Do not paste large arrays or base64 blobs into chat unless asked.
Prefer saving large artifacts (images, tables, logs) to files and returning a reference/path.

## Dependency discipline
- When introducing optional/heavy dependencies, ensure the app still starts without them:
  - prefer **lazy imports** inside the tool/function, or
  - conditionally register tools only if deps are installed.
- Keep imports fast at module import time (avoid importing heavy libs in global scope unless truly required).

## Definition of Done (DoD)
A task is “done” only if:
- The requested behavior is implemented.
- Tests (or a relevant smoke-check) were run and results reported.
- Docs/docstrings are updated if behavior or UX changed.
