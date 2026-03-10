#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep this quality scope focused on the packaged/release-facing surface until
# broader repo-wide lint/type debt is addressed separately.
TARGETS=(
  app.py
  main.py
  ts_agents/__init__.py
  ts_agents/config.py
  ts_agents/runtime_paths.py
  ts_agents/hosted_app.py
  tests/test_package_metadata.py
  tests/test_hosted_app.py
  tests/cli/test_entrypoints.py
)

uv run --with ruff ruff check "${TARGETS[@]}"
uv run --with mypy mypy \
  --ignore-missing-imports \
  --follow-imports skip \
  "${TARGETS[@]}"
