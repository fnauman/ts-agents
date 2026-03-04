# Sandbox execution

ts-agents supports multiple sandbox backends for tool execution:

- `local` (in-process)
- `subprocess` (separate Python process)
- `docker` (containerized local execution)
- `daytona` (managed cloud sandbox)
- `modal` (serverless execution)

## Quick start

```bash
uv run ts-agents run describe_series --param series=[1,2,3,4] --sandbox subprocess
```

Set a default backend:

```bash
export TS_AGENTS_SANDBOX_MODE=subprocess
```

## Docker sandbox

Build the image from repo root:

```bash
./build_docker_sandbox.sh ts-agents-sandbox:latest
```

This uses `Dockerfile.sandbox`.

Run with docker sandbox:

```bash
uv run ts-agents run describe_series --param series=[1,2,3,4] --sandbox docker
```

Useful env vars:
- `TS_AGENTS_DOCKER_IMAGE`
- `TS_AGENTS_DOCKER_CPUS`
- `TS_AGENTS_DOCKER_MEMORY_MB`
- `TS_AGENTS_DOCKER_TIMEOUT`
- `TS_AGENTS_DATA_DIR` (mounted read-only)

Use `--allow-network` if you need network access in docker mode.

## Daytona

Install and configure:

```bash
pip install daytona
export TS_AGENTS_SANDBOX_MODE=daytona
```

Then run any `ts-agents run ... --sandbox daytona` command.

## Modal

Install and authenticate:

```bash
pip install modal
modal setup
modal deploy src/sandbox/modal_app.py
```

Configure and run:

```bash
export TS_AGENTS_SANDBOX_MODE=modal
export TS_AGENTS_MODAL_APP=ts-agents-sandbox
export TS_AGENTS_MODAL_FUNCTION=run_tool
uv run ts-agents run describe_series --param series=[1,2,3,4] --sandbox modal
```
