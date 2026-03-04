#!/usr/bin/env bash
set -euo pipefail

# Build the Docker sandbox image used by SandboxMode.DOCKER.

IMAGE_NAME=${1:-ts-agents-sandbox:latest}

docker build -f Dockerfile.sandbox -t "${IMAGE_NAME}" .

echo "Built ${IMAGE_NAME}"
