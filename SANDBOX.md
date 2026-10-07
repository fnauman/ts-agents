# Sandbox execution

ts-agents supports multiple sandbox backends for tool execution:

- `local` (in-process)
- `subprocess` (separate Python process)
- `docker` (containerized local execution)
- `daytona` (managed cloud sandbox)
- `modal` (serverless execution)

## Quick start

```bash
uv run ts-agents sandbox list
uv run ts-agents sandbox doctor docker
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox subprocess
```

Set a default backend:

```bash
export TS_AGENTS_SANDBOX_MODE=subprocess
```

Fallback is now explicit. If a requested backend is unavailable, `ts-agents`
fails with a typed backend error unless you opt in:

```bash
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox docker
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox docker --allow-fallback
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox modal --fallback-backend local
```

## Sandbox environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TS_AGENTS_SANDBOX_MODE` | Default backend (`local`/`subprocess`/`docker`/`daytona`/`modal`) | `local` |
| `TS_AGENTS_DOCKER_IMAGE` | Docker image name for docker backend | `ts-agents-sandbox:latest` |
| `TS_AGENTS_DOCKER_CPUS` | CPU limit for docker backend | docker default |
| `TS_AGENTS_DOCKER_MEMORY_MB` | Memory limit for docker backend when the tool sets none | `2048` |
| `TS_AGENTS_DOCKER_TIMEOUT` | Timeout for docker backend commands | `300` |
| `TS_AGENTS_SUBPROCESS_TIMEOUT` | Timeout for subprocess backend commands | `300` |
| `DAYTONA_API_KEY` | Daytona auth token | _(none)_ |
| `DAYTONA_API_URL` | Daytona API endpoint override | Daytona default |
| `DAYTONA_TARGET` | Daytona target/region override | Daytona org default |
| `TS_AGENTS_DAYTONA_SNAPSHOT` | Daytona snapshot override | `daytonaio/sandbox:0.4.3` |
| `TS_AGENTS_DAYTONA_REPO_BRANCH` | Git branch cloned during Daytona bootstrap | default branch |
| `TS_AGENTS_DAYTONA_TIMEOUT` | Timeout for Daytona commands | `300` |
| `TS_AGENTS_DAYTONA_INSTALL_EXTRAS` | Comma-separated extras installed during Daytona bootstrap (`pip install -e .[extras]`); workflow runs set it automatically unless you set it; autoresearch runs always use the loop's `required_extras` | computed per run |
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
| `TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_FILE_BYTES` | Max single autoresearch artifact staged back from remote sandboxes; non-positive disables | `16777216` |
| `TS_AGENTS_AUTORESEARCH_ARTIFACT_MAX_TOTAL_BYTES` | Max total autoresearch artifact bytes staged back from remote sandboxes; non-positive disables | `67108864` |
| `TS_AGENTS_WORKFLOW_ARTIFACT_MAX_FILE_BYTES` | Max single workflow artifact staged back from remote sandboxes; non-positive disables | `16777216` |
| `TS_AGENTS_WORKFLOW_ARTIFACT_MAX_TOTAL_BYTES` | Max total workflow artifact bytes staged back from remote sandboxes; non-positive disables | `67108864` |

## Docker sandbox

Build the image from repo root:

```bash
./build_docker_sandbox.sh ts-agents-sandbox:latest
```

This uses `Dockerfile.sandbox` and installs the base profile. Pass optional
extras as a second argument (the `TS_AGENTS_EXTRAS` build argument):

```bash
bash build_docker_sandbox.sh ts-agents-sandbox:foundation foundation
# equivalent to:
docker build -f Dockerfile.sandbox --build-arg TS_AGENTS_EXTRAS=foundation \
  -t ts-agents-sandbox:foundation .
export TS_AGENTS_DOCKER_IMAGE=ts-agents-sandbox:foundation
```

Run with docker sandbox:

```bash
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox docker
```

Useful env vars:
- `TS_AGENTS_DOCKER_IMAGE`
- `TS_AGENTS_DOCKER_CPUS`
- `TS_AGENTS_DOCKER_MEMORY_MB`
- `TS_AGENTS_DOCKER_TIMEOUT`
- `TS_AGENTS_DATA_DIR` (mounted read-only)

Use `--allow-network` if you need network access in docker mode.

## Optional extras and foundation models

Optional families (`ml`, `neural`, `foundation`, `viz`, ...) must be installed
inside the sandbox, not only on the host. The `local` and `subprocess` backends
check the host environment before running; Docker, Daytona and Modal report a
missing extra from inside the sandbox, with the same install hint.

- **Daytona:** workflow runs set `TS_AGENTS_DAYTONA_INSTALL_EXTRAS`
  automatically from the workflow's required extras, the extras of the
  requested methods, and `viz` unless `--skip-plots` is passed. For example
  `forecast-panel --methods lightgbm,chronos2` installs `foundation,ml,viz`.
  This applies to every workflow, including `inspect-series` and
  `activity-recognition`. For workflow runs, a value you set yourself is never
  overridden. Autoresearch runs always install the loop's `required_extras`
  (for example `foundation` for `foundation-smoke`).
- **Docker:** the default image is base-only; build an extras image as shown
  above. Containers run with `--network none` unless you pass
  `--allow-network`, and the image has no foundation-model weights baked in.
  The backend also uses a read-only root filesystem and mounts only the I/O
  directory and `TS_AGENTS_DATA_DIR`, so there is no writable or mounted
  Hugging Face cache today. Foundation-model methods therefore need a manual
  `docker run` with a mounted, pre-populated `HF_HOME` and `HF_HUB_OFFLINE=1`
  (or network access plus a writable cache). The `local` and `subprocess`
  backends use the host cache.
- **Modal:** the deployed image installs the base dependencies from
  `pyproject.toml` only; optional extras are not installed there yet.
- **Weights:** foundation models download only the requested checkpoints into
  `HF_HOME` on first use (roughly 0.1-1 GB each). A checkpoint that is neither
  cached nor downloadable fails with `FoundationModelUnavailableError`, reported
  in the JSON envelope as `backend_unavailable` (exit code 5) with the
  remediation in `hint`.
- **Resources:** one foundation model needs a few GB of RAM, and a run keeps
  every requested model loaded until it finishes. In memory-limited sandboxes,
  request one model at a time. The Docker backend applies
  `TS_AGENTS_DOCKER_MEMORY_MB` (default 2048) and the `TS_AGENTS_*_TIMEOUT`
  defaults (300 s) when no tool-specific limit applies; raise them for
  foundation-model or panel runs.

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
# Optional branch override for testing an unmerged branch:
# export TS_AGENTS_DAYTONA_REPO_BRANCH=my-feature-branch
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
Then run any `ts-agents tool run ... --sandbox daytona`, `ts-agents workflow run ... --sandbox daytona`, or `ts-agents autoresearch run ... --sandbox daytona` command.

Daytona-oriented autoresearch smoke examples (`--max-trials` counts model/evaluation-spec rows):

```bash
uv run ts-agents autoresearch run forecast-daytona \
  --profile smoke \
  --models seasonal_naive \
  --sandbox daytona \
  --json

uv run ts-agents autoresearch run classify-daytona \
  --profile smoke \
  --dataset synthetic \
  --models knn \
  --sandbox daytona \
  --json
```

## Modal

The current Modal deploy module is a source-checkout path. It builds the Modal
image from the repo's `pyproject.toml` and local `ts_agents/` tree, so deploy
it from a git checkout rather than from an installed wheel.

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

# Deploy the app in a specific Modal environment (recommended, from repo root)
modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox
```

Configure and run:

```bash
export TS_AGENTS_SANDBOX_MODE=modal
export MODAL_ENVIRONMENT=main
export TS_AGENTS_MODAL_APP=ts-agents-sandbox
export TS_AGENTS_MODAL_FUNCTION=run_tool
uv run ts-agents tool run describe_series --input-json '{"series":[1,2,3,4]}' --sandbox modal
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
uv run ts-agents tool run stl_decompose_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --sandbox daytona
```

Modal smoke test:

```bash
# First deploy once after auth, from the repo root:
uv run modal deploy -m ts_agents.sandbox.modal_app --env main --name ts-agents-sandbox

uv run ts-agents tool run stl_decompose_with_data \
  --run Re200Rm200 \
  --var bx001_real \
  --sandbox modal
```
