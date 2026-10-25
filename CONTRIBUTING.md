# Contributing

## Setup

```bash
git clone https://github.com/fnauman/ts-agents.git
cd ts-agents
uv sync --frozen --python 3.12
uv run --no-sync ts-agents --help
```

Plain `uv sync` installs the base CLI and developer tools, not every analysis
backend. Use Python 3.11–3.13 for the locked optional stack; installed base-wheel
smokes also cover 3.14. After selecting a profile, `uv run --no-sync` preserves
it rather than changing dependencies as part of a check.

## Choose a check profile

Start with the affected existing tests. These quick base-contract checks need
no model weights, training, network services, UI, or optional analysis backends:

```bash
uv run --no-sync python -m pytest -q \
  tests/cli/test_input_parsing.py tests/cli/test_output.py \
  tests/cli/test_skills.py tests/core/test_core_imports.py \
  tests/test_package_metadata.py
```

For statistical forecasting/decomposition changes, install those backends and
run the relevant existing modules:

```bash
uv sync --frozen --python 3.12 --extra forecasting --extra decomposition --extra viz
uv run --no-sync python -m pytest -q \
  tests/core/test_forecasting.py tests/core/test_decomposition.py \
  tests/core/test_comparison.py
```

Panel, classification, external tool-wrapper, and pattern work need their
respective `ml`/`neural`, `classification`, `agents`, or `patterns` extras.
Select only the required extras and tests. `agents` supplies LangChain tool
wrappers for external runtimes; UI and built-in agents were removed in 0.4.0.
See the README for environment settings and sandbox commands.

The full suite requires the complete optional profile. CI qualifies this
locked stack on Python 3.11–3.13:

```bash
uv sync --frozen --all-extras --python 3.12
uv run --no-sync python -m pytest -q
bash scripts/check_release_quality.sh
git diff --check
```

Real foundation-model checks are opt-in via
`TS_AGENTS_RUN_FOUNDATION_TESTS=1` and need cached or downloadable weights.
They are not part of ordinary PR qualification. Do not enable downloads,
training campaigns, GPUs, or cloud services implicitly during routine checks.
Release qualification additionally verifies installed artifacts; see
[Distribution](docs/distribution.qmd).

## Development workflow

1. Create an issue and branch for the task. State the behavior, compatibility
   boundaries, exclusions, and validation budget before nontrivial changes.
2. Implement the smallest complete fix. Prefer extending an existing test
   with a real regression or independent numerical expectation.
3. Update the existing documentation affected by the change. For skills, sync
   the full canonical directory to `ts_agents/resources/skills/`, including
   supporting files; `tests/cli/test_skills.py` checks mirror integrity.
4. Run the appropriate checks above. Report commands, environment, pass/fail
   results, unsupported paths, retained test-case changes, and runtime impact.
5. Open a scoped PR using the template. Include documentation, manifest/profile
   impacts, material risks, and follow-ups. Preserve compatibility or explain
   intentional changes.

Ordinary CI runs once per PR revision and on main, cancels superseded PR runs,
and shares one verified wheel across the base-install smoke matrix. Release
qualification remains separate on version tags. Jobs have explicit timeouts;
passing selected checks does not qualify unexercised GPUs, remote backends,
external agent interactions, or every declared dependency minimum.

## Issue reports

Include expected and actual behavior, a minimal reproduction, the command,
Python/package versions, execution backend, and relevant warnings/artifacts.
Avoid posting credentials or sensitive data.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
