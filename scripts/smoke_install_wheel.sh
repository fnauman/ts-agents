#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash scripts/smoke_install_wheel.sh <python-version> [dist-dir-or-wheel]" >&2
  exit 2
fi

PYTHON_VERSION="$1"
TARGET="${2:-dist}"

resolve_wheel() {
  local target="$1"
  local whl
  local wheels=()
  if [ -f "$target" ]; then
    printf '%s\n' "$target"
    return 0
  fi

  while IFS= read -r whl; do
    wheels+=("$whl")
  done < <(find "$target" -maxdepth 1 -type f -name 'ts_agents-*.whl' | sort)

  if [ "${#wheels[@]}" -ne 1 ]; then
    echo "Expected exactly 1 wheel in $target, found ${#wheels[@]}" >&2
    printf '%s\n' "${wheels[@]}" >&2
    exit 1
  fi

  printf '%s\n' "${wheels[0]}"
}

WHEEL="$(resolve_wheel "$TARGET")"
VENV="$(mktemp -d "/tmp/ts-agents-release-smoke-${PYTHON_VERSION//./}-XXXXXX")"
trap 'rm -rf "$VENV"' EXIT

uv venv "$VENV" --python "$PYTHON_VERSION"
uv pip install --python "$VENV/bin/python" "$WHEEL"

# Run entrypoints outside the checkout, including the subprocess jobs launch.
cd "$VENV"

"$VENV/bin/ts-agents" --help >/dev/null
"$VENV/bin/ts-agents" tool list --bundle demo >/dev/null
"$VENV/bin/ts-agents-ui" --help >/dev/null
# -I keeps the repo checkout off sys.path so the installed wheel is what
# actually gets imported and asserted against.
"$VENV/bin/python" -I - <<'PY'
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import ts_agents
import ts_agents.hosted_app as hosted
from ts_agents.runtime_paths import resolve_default_data_dir, resolve_default_skills_dir

assert ts_agents.__version__ == version("ts-agents")
assert resolve_default_data_dir().exists()
assert resolve_default_skills_dir().exists()

# The base wheel has no gradio; the hosted app must fail with an actionable
# extras hint rather than a bare ModuleNotFoundError.
try:
    hosted.app
except ImportError as exc:
    assert "ts-agents[ui]" in str(exc), exc
else:
    raise AssertionError("hosted.app should require the ui extra in a base install")

with tempfile.TemporaryDirectory(prefix="wheel-cli-", dir=Path.cwd()) as directory:
    workdir = Path(directory)

    def invoke(*args):
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "ts_agents", *args],
            cwd=workdir, capture_output=True, text=True, timeout=30,
            env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}},
        )
        assert completed.returncode == 0, (args, completed.stdout, completed.stderr)
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True and payload["schema_version"] == "1.0", payload
        return payload["result"]

    capabilities = invoke("capabilities", "--json")
    assert capabilities["background_jobs"]["available"] is True
    invoke("skills", "validate", "--json")
    series = '{"series":[1,2,3,4,5,6,7,8,9,10,11,12]}'
    inspected = invoke("workflow", "run", "inspect-series", "--input-json", series, "--skip-plots", "--json")
    forecast_args = ["workflow", "run", "forecast-series", "--input-json", series,
                     "--horizon", "3", "--methods", "seasonal_naive", "--skip-plots", "--json"]
    forecast = invoke(*forecast_args)
    for workflow in (inspected, forecast):
        for artifact in workflow["artifacts"]:
            assert Path(artifact["path"]).is_absolute() and Path(artifact["path"]).is_file(), artifact
        manifest = json.loads(Path(workflow["data"]["manifest_path"]).read_text())
        assert manifest["resume_identity"]["input_sha256"]
    invoke("autoresearch", "run", "forecast-daytona", "--profile", "smoke", "--models", "seasonal_naive", "--json")
    catalog = invoke("runs", "list", "--json")
    assert {"inspect-series", "forecast-series", "forecast-daytona"} <= {run["name"] for run in catalog["runs"]}
    invoke("runs", "show", forecast["data"]["run_id"], "--json")
    preview = invoke("runs", "gc", "--older-than", "30", "--json")
    assert preview["dry_run"] is True and preview["matched"] == 0

    job = invoke("jobs", "start", "--json", "--", *forecast_args)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = invoke("jobs", "status", job["job_id"], "--json")
        if status["status"] in {"completed", "failed", "cancelled", "stale"}:
            break
        time.sleep(0.1)
    assert status["status"] == "completed" and status["exit_code"] == 0, status
    logs = invoke("jobs", "logs", job["job_id"], "--json")
    assert "forecast-series" in "\n".join(logs["lines"]), logs
print("installed wheel smoke ok")
PY
