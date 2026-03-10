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

"$VENV/bin/ts-agents" --help >/dev/null
"$VENV/bin/ts-agents" tool list --bundle demo >/dev/null
"$VENV/bin/ts-agents-ui" --help >/dev/null
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

import ts_agents
import ts_agents.hosted_app as hosted
from ts_agents.runtime_paths import resolve_default_data_dir, resolve_default_skills_dir

assert ts_agents.__version__ == version("ts-agents")
assert resolve_default_data_dir().exists()
assert resolve_default_skills_dir().exists()
assert hosted.app is not None
print("installed wheel smoke ok")
PY
