# Contributing

Thanks for helping improve `ts-agents`.

## Quick setup

```bash
git clone https://github.com/fnauman/ts-agents.git
cd ts-agents
uv sync
```

## Development workflow

1. Create an issue describing the bug/feature before large changes.
2. Create a branch from `main`.
3. Implement the change with focused commits.
4. Run relevant checks locally:
   - `uv run python -m pytest -q`
   - optionally targeted tests such as `uv run python -m pytest -q tests/cli`
5. Open a PR to `main` with:
   - behavior changes
   - test coverage added/updated
   - risks or follow-up items

## PR quality expectations

- Keep PRs scoped to one logical change.
- Add or update tests when behavior changes.
- Update docs (`README.md`, demo docs, or module docs) when user-facing behavior changes.
- Preserve backward compatibility where practical; if breaking behavior is intentional, call it out clearly in the PR.

## Issue reports

Please include:

- expected behavior
- actual behavior
- minimal repro steps
- relevant environment details (OS, Python version, command used)

## Community standards

By participating, you agree to follow the project [Code of Conduct](CODE_OF_CONDUCT.md).
