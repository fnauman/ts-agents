# AGENTS.md

This repo is a time-series analysis toolkit + Gradio app with both:
- **Manual analysis tabs** (decomposition / forecasting / patterns / classification)
- **Agent chat** (simple + deep multi-agent)

The goal of this file is to make coding agents effective and safe in this codebase.

## Quick commands (copy/paste)

### Setup
```bash
uv sync
```

### Run the app
```bash
uv run python main.py
```

Common options (see `main.py --help`):
```bash
uv run python main.py --agent-type deep
uv run python main.py --no-agent
uv run python main.py --share
uv run python main.py --port 8080
```

### Run tests
```bash
uv run python -m pytest -q
```

### Environment variables
- `TS_AGENTS_DATA_DIR` (optional): path to the dataset directory (see `src/config.py`)
- `OPENAI_MODEL` (optional): defaults to `gpt-4o-mini` (see `src/config.py`)
- `OPENAI_API_KEY` (required for agent chat in many setups)

## Repo map (where to look first)
- `main.py` — app entrypoint (arg parsing + launch)
- `src/ui/` — Gradio UI (tabs + chat)
- `src/agents/` — agent implementations (simple + deep)
- `src/tools/` — tool registry + wrappers (LangChain + deep agent tools)
- `src/core/` — pure analysis implementations (decomposition, forecasting, patterns, classification, complexity, spectral)
- `src/persistence/` — session persistence + caching
- `tests/` — unit tests

## Working agreement (always follow)

### GitHub workflow & issue tracking
- Use the `gh` CLI.
- Create an issue + branch per task before coding.
- Update the issue with each incremental step.
- If blocked: document why + next action.
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
