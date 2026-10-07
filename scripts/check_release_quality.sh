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
  ts_agents/cli/runs.py
  ts_agents/cli/jobs.py
  ts_agents/cli/jobs_worker.py
  ts_agents/tools/artifact_staging.py
  scripts/verify_release_artifacts.py
  ts_agents/workflows/lifecycle.py
  ts_agents/core/forecasting/catalog.py
  ts_agents/core/forecasting/foundation.py
  ts_agents/core/forecasting/panel.py
  ts_agents/workflows/panel.py
  tests/core/test_panel_forecasting.py
  tests/core/test_forecasting_catalog.py
  tests/core/test_foundation_adapter.py
  tests/core/test_foundation_real.py
  tests/cli/test_data_export.py
  tests/cli/test_runs.py
  tests/cli/test_jobs.py
  tests/test_package_metadata.py
  tests/test_hosted_app.py
  tests/cli/test_entrypoints.py
)

uv run ruff check "${TARGETS[@]}"
uv run mypy \
  --ignore-missing-imports \
  --follow-imports skip \
  "${TARGETS[@]}"
