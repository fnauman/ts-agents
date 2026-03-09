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

## Sandbox environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TS_AGENTS_SANDBOX_MODE` | Default backend (`local`/`subprocess`/`docker`/`daytona`/`modal`) | `local` |
| `TS_AGENTS_DOCKER_IMAGE` | Docker image name for docker backend | `ts-agents-sandbox:latest` |
| `TS_AGENTS_DOCKER_CPUS` | CPU limit for docker backend | docker default |
| `TS_AGENTS_DOCKER_MEMORY_MB` | Memory limit for docker backend | docker default |
| `TS_AGENTS_DOCKER_TIMEOUT` | Timeout for docker backend commands | `300` |
| `DAYTONA_API_KEY` | Daytona auth token | _(none)_ |
| `DAYTONA_API_URL` | Daytona API endpoint override | Daytona default |
| `DAYTONA_TARGET` | Daytona target/region override | Daytona org default |
| `TS_AGENTS_DAYTONA_SNAPSHOT` | Daytona snapshot override | `daytonaio/sandbox:0.4.3` |
| `TS_AGENTS_DAYTONA_TIMEOUT` | Timeout for Daytona commands | `300` |
| `TS_AGENTS_DAYTONA_STREAM` | Stream Daytona bootstrap/runner logs to stderr | `true` |
| `TS_AGENTS_DAYTONA_LOG_FILE` | Append Daytona streamed logs to file | _(none)_ |
| `MODAL_TOKEN_ID` | Modal token id (headless/CI auth) | profile config |
| `MODAL_TOKEN_SECRET` | Modal token secret (headless/CI auth) | profile config |
| `MODAL_ENVIRONMENT` | Modal environment used for deploy/lookup | profile default |
| `TS_AGENTS_MODAL_APP` | Modal app name for remote execution | `ts-agents-sandbox` |
| `TS_AGENTS_MODAL_FUNCTION` | Modal function name for remote execution | `run_tool` |
| `TS_AGENTS_MODAL_STREAM` | Stream `modal app logs` while remote call runs | `true` |
| `TS_AGENTS_MODAL_LOG_FILE` | Append Modal stream logs to file | _(none)_ |
| `TS_AGENTS_MODAL_LOG_TIMESTAMPS` | Include timestamps in Modal log stream | `false` |

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
export DAYTONA_API_KEY=your_daytona_api_key
# Optional Daytona endpoint overrides:
# export DAYTONA_API_URL=...
# export DAYTONA_TARGET=...
# Optional snapshot override:
# export TS_AGENTS_DAYTONA_SNAPSHOT=daytonaio/sandbox:0.4.3
# Optional: stream bootstrap/runner logs to stderr (default=true)
# export TS_AGENTS_DAYTONA_STREAM=true
# Optional: persist streamed Daytona logs to a local file
# export TS_AGENTS_DAYTONA_LOG_FILE=outputs/logs/daytona.log
export TS_AGENTS_SANDBOX_MODE=daytona
```

`ts-agents` now bootstraps Daytona sandboxes by default by cloning
`https://github.com/fnauman/ts-agents` into `workspace/ts-agents` and running
`pip install -e` there before tool execution. The default snapshot is
`daytonaio/sandbox:0.4.3`; override with `TS_AGENTS_DAYTONA_SNAPSHOT` if needed.

Then run any `ts-agents run ... --sandbox daytona` command.

## Modal

Install and authenticate (Modal uses token id/secret, not a single API key):

```bash
pip install modal
# Interactive auth (creates token and opens browser login)
modal token new

# Or headless/CI auth
# modal token set --token-id <id> --token-secret <secret>
# export MODAL_TOKEN_ID=<id>
# export MODAL_TOKEN_SECRET=<secret>
# Optional: persist Modal stream logs to file
# export TS_AGENTS_MODAL_LOG_FILE=outputs/logs/modal.log
# Optional: disable/enable log streaming and timestamps
# export TS_AGENTS_MODAL_STREAM=true
# export TS_AGENTS_MODAL_LOG_TIMESTAMPS=false

# Verify auth
modal token info

# Deploy the app in a specific Modal environment (recommended)
modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox
```

Configure and run:

```bash
export TS_AGENTS_SANDBOX_MODE=modal
export MODAL_ENVIRONMENT=main
export TS_AGENTS_MODAL_APP=ts-agents-sandbox
export TS_AGENTS_MODAL_FUNCTION=run_tool
uv run ts-agents run describe_series --param series=[1,2,3,4] --sandbox modal
```

If you see:
`App 'ts-agents-sandbox' not found in environment 'main'`

- Check deployment visibility: `modal app list --env main`
- Re-deploy explicitly into the same environment:
  `modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox`
- Ensure runtime lookup matches deploy target:
  `export MODAL_ENVIRONMENT=main`

## Cloud sandbox smoke tests

Run these after auth and deployment setup to verify end-to-end execution.

Daytona smoke test:

```bash
uv run ts-agents run stl_decompose_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --sandbox daytona
```

Modal smoke test:

```bash
# First deploy once after auth:
uv run modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox

uv run ts-agents run stl_decompose_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --sandbox modal
```
