#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -rf "$ROOT/build" "$ROOT/dist" "$ROOT/wheels"
find "$ROOT" -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf {} +

echo "release artifacts cleaned"
