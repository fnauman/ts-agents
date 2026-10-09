#!/usr/bin/env bash
set -euo pipefail

# Build the Docker sandbox image used by SandboxMode.DOCKER.
#
# Usage: bash build_docker_sandbox.sh [image-name] [extras]
#   extras: optional comma-separated ts-agents extras, e.g. "forecasting,ml" or
#   "foundation". The default image is the base install.

IMAGE_NAME=${1:-ts-agents-sandbox:latest}
EXTRAS=${2:-}

docker build -f Dockerfile.sandbox --build-arg "TS_AGENTS_EXTRAS=${EXTRAS}" -t "${IMAGE_NAME}" .

echo "Built ${IMAGE_NAME}${EXTRAS:+ with extras [${EXTRAS}]}"
