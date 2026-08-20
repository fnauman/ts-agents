# Roadmap

This roadmap is intentionally lightweight and outcome-focused.
It complements `README.md` (what exists today) with direction (what is next).

## Working Principles

- Keep the `ts-agents` CLI contract stable.
- Treat artifacts (plots, reports, JSON, logs) as first-class outputs.
- Keep agent frameworks and UI layers swappable.
- Use sandbox backends to handle dependency and runtime isolation.

## Recently Shipped (0.2.0)

- Run catalog: `ts-agents runs list/show/gc` over every `run_manifest.json`.
- Background jobs: `ts-agents jobs start/list/status/logs/cancel` with durable
  job records, log capture, and process-group cancellation.
- Python 3.14 base-install support; CI wheel smoke on 3.14.
- Weekly unmocked `foundation-chronos-smoke` CI run (real chronos + torch),
  so chronos API drift is caught in CI rather than by users.

## Current Focus

- Publish 0.2.0 to PyPI (tag `v0.2.0`; the publish workflow does the rest).
  PyPI has been frozen at 0.1.1 since March — releasing is the highest-leverage
  single action.
- Resumption for autoresearch loops (`--resume` exists for workflows only).
- Job-aware progress reporting: stream trial/step progress into the job record
  so `jobs status` can show percent-complete for long runs.

## Decisions Needed

- **agents/ + ui/ (~20% of the package):** the deep-agent adapter and Gradio UI
  are labeled experimental and carry no real test coverage. Decide within the
  0.3.x cycle: extract to a separate repo, delete, or commit with real tests.
  Until then, no polish-only PRs against these layers.
- **MCP:** the ecosystem converged on MCP for tool discovery/contracts. A thin
  MCP server over the existing registry would be cheap and meet agent harnesses
  where they are. Decide deliberately whether "CLI+skills only" is the bet, and
  record why.

## Next

- Better environment resolution and caching for heavy optional dependencies
  (numba-based extras still block full 3.14 support).
- Richer experiment history and diffing of run outputs/artifacts on top of the
  `runs` catalog.

## Later

- Human-in-the-loop gates for expensive/high-risk operations.
- Improved hybrid tool routing (heuristics + LLM + evaluation feedback).

## How This Roadmap Is Maintained

- Keep entries short and measurable.
- Prefer capability-level milestones over date promises.
- Update whenever a major direction changes or a milestone lands.
