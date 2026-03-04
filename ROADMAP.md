# Roadmap

This roadmap is intentionally lightweight and outcome-focused.
It complements `README.md` (what exists today) with direction (what is next).

## Working Principles

- Keep the `ts-agents` CLI contract stable.
- Treat artifacts (plots, reports, JSON, logs) as first-class outputs.
- Keep agent frameworks and UI layers swappable.
- Use sandbox backends to handle dependency and runtime isolation.

## Current Focus

- Improve long-running execution ergonomics: progress, cancellation, and resumption.
- Strengthen dependency isolation and reproducibility across backends.
- Expand benchmark/evaluation workflows for tool routing and bundle quality.

## Next

- Async-first runtime model for expensive tasks (queue/job semantics).
- Better environment resolution and caching for heavy optional dependencies.
- Richer experiment history and diffing of run outputs/artifacts.

## Later

- UI workbench upgrades: artifact browser, run monitor, and review-oriented UX.
- Human-in-the-loop gates for expensive/high-risk operations.
- Improved hybrid tool routing (heuristics + LLM + evaluation feedback).

## How This Roadmap Is Maintained

- Keep entries short and measurable.
- Prefer capability-level milestones over date promises.
- Update whenever a major direction changes or a milestone lands.
