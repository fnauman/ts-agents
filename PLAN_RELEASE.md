# PyPI Release Plan for `ts-agents`

## Goal

Prepare the first public PyPI release of `ts-agents` from the current repo state.

This document synthesizes:
- the PyPI release-readiness review performed on 2026-03-09
- the supplied release plan and checklist
- current build and install validation results from this repo

The point is not just "can `uv build` produce artifacts", but "will a user who installs the published package from outside the repo get a correct, documented, supportable experience."

---

## Current Validation Snapshot

Checks already run during review:

- `uv build`
  - succeeded for both sdist and wheel
- `uv run python -m pytest -q`
  - `345 passed in 19.88s`
- isolated wheel install from `/tmp` on Python 3.12
  - succeeded
  - `python -m ts_agents --help` worked
  - packaged resource paths resolved correctly
- isolated wheel install from `/tmp` on Python 3.14
  - failed during dependency resolution
  - failure path was `ts-agents -> neuralforecast -> ray`, where no compatible `cp314` wheel was available

Artifact inspection also confirmed:

- the built wheel contains `ts_agents/resources/...`
- the built wheel does **not** contain repo-root files such as `app.py`, `main.py`, `docs/`, or `data/wisdm_subset.csv`

---

## Overall Verdict

`ts-agents` is **not ready** for the first PyPI release yet.

The repo is close in the sense that:
- packaging metadata exists
- the publish workflow exists
- the project builds cleanly
- the tests pass in the source checkout

But the first release should be blocked until the PyPI artifact surface is corrected.

Current priority split:

- `P0` must complete before release: 6 items
- `P1` medium priority, should do before release if possible: 7 items
- `P2` low/optional polish: 5 items

---

## P0: Must Complete Before Release

### P0.1 Fix the advertised Python support range

**Why this is blocking**

`pyproject.toml` currently advertises `requires-python = ">=3.11"`, which includes Python 3.14. In practice, isolated installation on Python 3.14 currently fails because `neuralforecast` pulls `ray`, and the resolved dependency set is not installable there.

**Required outcome**

Choose one:

- cap support to a version range that is actually installable, for example `<3.14`
- make the problematic dependency stack optional via extras
- or validate and pin a dependency set that truly works on every advertised Python version

**Files**

- `pyproject.toml`
- CI/release validation if the supported matrix changes
- `README.md` and docs if supported Python versions are described there

**Done when**

Every Python version advertised in package metadata can install the built wheel in a clean environment outside the repo.

### P0.2 Make the PyPI README safe for PyPI rendering

**Why this is blocking**

The current README uses repo-relative links and images. Those are acceptable on GitHub but not reliable on the PyPI project page. The current README also mixes source-checkout instructions with installed-package instructions.

**Required outcome**

- replace repo-relative links and media with absolute GitHub Pages or GitHub blob/raw URLs where appropriate
- or split the README into a PyPI-safe long description and a fuller GitHub README
- ensure the PyPI-facing install section describes the installed package path, not just `uv sync` from a checkout

**Files**

- `README.md`
- optionally a separate release README if you choose a split-doc approach

**Done when**

- `twine check dist/*` passes
- the README renders correctly on TestPyPI
- every command on the PyPI project page is valid for users who installed from PyPI

### P0.3 Align the documented public entrypoints with what the wheel actually installs

**Why this is blocking**

The wheel installs the `ts_agents` package and the `ts-agents` console script, but it does not include repo-root helpers like `app.py` and `main.py`. The docs currently tell users to run those repo-root files.

**Required outcome**

Choose one:

- provide supported installed-package entrypoints for the UI and hosted app
- or stop documenting repo-root entrypoints on the PyPI path

Also fix the sdist/test mismatch:

- either include the required hosted-app file(s) in the sdist
- or stop shipping tests in the sdist that import repo-root files not present there

**Files**

- `README.md`
- `docs/huggingface-spaces.qmd`
- `tests/test_hosted_app.py`
- packaging config if additional entrypoints or files are added

**Done when**

Every documented public entrypoint works from a clean installed-package environment, not only from a source checkout.

### P0.4 Fix packaged demo/data claims

**Why this is blocking**

The docs and README currently present `data/wisdm_subset.csv` as checked-in and usable, but that file is not present in the built wheel. That means the documented WISDM demo path is not actually part of the PyPI package experience.

**Required outcome**

Choose one:

- package the WISDM subset and any other advertised repo-root demo assets
- or stop describing them as bundled / built-in for PyPI users
- clearly separate "from repo checkout" instructions from "from installed package" instructions

**Files**

- `README.md`
- `docs/walkthroughs.qmd`
- `ts_agents/resources/data/README.md`
- packaging config and package data if the asset is added

**Done when**

Every data file described as bundled or built-in is actually present in the built artifact and usable from a clean wheel install.

### P0.5 Declare direct runtime dependencies explicitly

**Why this is blocking**

Shipped modules import direct runtime dependencies that are not declared in `pyproject.toml`. Today this works only because those packages arrive transitively from other dependencies. That is fragile for a first public release.

Known direct imports to fix:

- `scipy`
- `pydantic`

**Required outcome**

- add missing direct runtime dependencies to `pyproject.toml`
- perform a quick import audit for similar transitive-only assumptions

**Files**

- `pyproject.toml`
- optionally tests if you want explicit import smoke coverage

**Done when**

The package no longer depends on undeclared transitive dependencies for its shipped import surface.

### P0.6 Add release-gate validation for the built artifact

**Why this is blocking**

The current CI and publish workflow prove that the repo builds and tests from a checkout, but they do not prove that the built wheel installs cleanly or that the long description will render correctly on PyPI.

**Required outcome**

Add at least:

- `twine check dist/*`
- wheel install smoke test outside the repo root
- console script smoke test from the installed wheel
- for the first release, a TestPyPI validation step before the real publish

**Files**

- `.github/workflows/ci.yml`
- `.github/workflows/publish-pypi.yml`
- optionally `docs/distribution.qmd`

**Done when**

Publishing cannot proceed without artifact-level validation that matches the PyPI user path.

---

## P1: Medium Priority

These are not the first-order blockers above, but they are good release work and should be done before release if time permits.

### P1.1 Add `__version__` to `ts_agents/__init__.py`

This is a good package ergonomics improvement for `v0.1.0`. It is not as critical as install correctness, but it is standard and useful.

Recommended implementation:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ts-agents")
except PackageNotFoundError:
    __version__ = "0.1.0"
```

### P1.2 Add missing classifiers

The package should include stronger PyPI classifiers, especially:

- `License :: OSI Approved :: MIT License`
- `Intended Audience :: Science/Research`
- `Intended Audience :: Developers`

This improves PyPI metadata quality but does not fix runtime correctness.

### P1.3 Add `py.typed`

If the project intends to present itself as shipping inline type hints, add:

- `ts_agents/py.typed`
- package-data inclusion for `py.typed`

This is useful for downstream users and is a clean `0.1.0` improvement.

### P1.4 Resolve the Modal backend packaging assumption

`ts_agents/sandbox/modal_app.py` currently assumes a source checkout layout and reaches for a neighboring `pyproject.toml`. That is fine for repo-local workflows, but not for an installed package story.

Choose one:

- fix the backend so it can be driven from the installed package
- or clearly document it as source-checkout-only and remove/qualify any broader claim

If Modal remains a prominently advertised backend for the release, promote this item to `P0`.

### P1.5 Reduce or at least document the heavy default dependency footprint

The isolated Python 3.12 install pulled a very large stack, including heavy ML/runtime dependencies. That may be intentional, but if not, the first release is the right time to split extras such as:

- core CLI/data analysis
- UI
- agent stack
- heavy forecasting backends
- cloud/sandbox backends

If you keep the current all-in install, document the expected footprint honestly.

### P1.6 Clean up import-time configuration side effects

`ts_agents/config.py` currently reads `~/.env`, mutates `os.environ`, and binds some defaults from `Path.cwd()` at import time. That is surprising behavior for a package import.

This is not necessarily a release blocker, but it is worth tightening before publishing a package others may import programmatically.

### P1.7 Align hosted deployment docs with actual dependency sources

The Hugging Face Spaces docs currently point at `requirements.txt`, which has already drifted from `pyproject.toml` by including `pytest`. Decide which file is authoritative for deployment and keep the docs aligned.

---

## P2: Low Priority / Optional Polish

These are useful cleanups, but they should not block release once `P0` is complete.

### P2.1 Add author email to package metadata

Good PyPI hygiene, but optional.

### P2.2 Review dependency lower bounds

Several lower bounds are very recent. That is acceptable for an alpha, but it narrows the install base. Revisit only if broader compatibility matters for the first release.

### P2.3 Add linting and type-checking to CI

`ruff`, `mypy`, or equivalent would improve confidence, but they are not prerequisites for the first publish if install correctness and tests are already covered.

### P2.4 Remove stale `.gitignore` rules

The `!src/resources/...` exceptions are stale cleanup from an older layout. Harmless, but worth removing.

### P2.5 Clean local build artifacts before any manual upload testing

`dist/` and `*.egg-info` are correctly ignored. This is only local housekeeping so manual testing does not accidentally use stale artifacts.

---

## Reclassification of the Supplied Plan

The supplied plan had several good items; they just need to be reprioritized against the concrete artifact-level failures.

| Supplied item | Reclassified priority | Notes |
|---|---|---|
| Add `__version__` | `P1` | Good package ergonomics, not a hard blocker |
| Add license/audience classifiers | `P1` | Good metadata quality |
| Clean `dist/` and `*.egg-info` locally | `P2` | Housekeeping only |
| Remove stale `.gitignore` lines | `P2` | Cleanup only |
| Add `py.typed` | `P1` | Useful if inline typing is intentional |
| Fresh wheel install test | `P0` | Must be part of release gating |
| TestPyPI first | `P0` | Strongly recommended for the first publish |
| Update README to mention `pip install ts-agents` after publish | `P0` | Required as part of PyPI-safe docs |
| Revisit very recent dependency minimums | `P2` | Compatibility/polish decision |
| Add lint/type-check CI | `P2` | Quality improvement, not a publish blocker |

---

## Recommended Implementation Order

1. Fix the Python support story in `pyproject.toml`.
2. Add missing direct runtime dependencies and run a clean wheel install smoke test.
3. Make README and docs safe for the installed-package path.
4. Decide whether WISDM and hosted/UI repo-root flows are packaged or repo-only, then align packaging and docs.
5. Add artifact-level CI/release gates: wheel install smoke test, console-script smoke test, `twine check`.
6. Run TestPyPI validation.
7. Apply medium-priority metadata and package-quality improvements.
8. Tag and publish only after all `P0` items are green.

---

## Release Gate Checklist

Before creating the first release tag:

1. `uv build`
2. `uv run python -m pytest -q`
3. `uv run python -c "import ts_agents; print(getattr(ts_agents, '__version__', 'missing'))"`
4. install the built wheel in a clean environment outside the repo root
5. run `ts-agents --help` from that installed wheel
6. run at least one documented command from the installed wheel path
7. run `twine check dist/*`
8. upload to TestPyPI and inspect the rendered page
9. confirm every advertised Python version can install the artifact
10. only then create the release tag that triggers the publish workflow

---

## Definition of Release-Ready

The repo is release-ready when all of the following are true:

- the built artifact installs cleanly on every advertised Python version
- the PyPI page is accurate and fully readable
- every documented "bundled" file or entrypoint is actually present in the built package
- the package declares the dependencies it imports directly
- the publish path validates the artifact before attempting a real upload

Until then, the correct status is:

**Packaging infrastructure exists, but the first public PyPI release should remain blocked.**
