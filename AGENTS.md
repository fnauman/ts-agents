# AGENTS.md

`ts-agents` is a CLI-first time-series toolkit. Workflows and tools are the
stable automation surface; external agents consume them through CLI or tool wrappers.
Setup, install profiles, and verification commands live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Scope and execution

1. **Explore:** read the affected code and authoritative documentation; state
   the current behavior and material uncertainty.
2. **Plan:** give numbered implementation and validation steps. State intended
   behavior, compatibility boundaries, exclusions, and a verification budget.
3. **Implement:** use the smallest complete change. Nearby cleanup and new
   abstractions require a concrete reason within the task.
4. **Verify:** run the smallest relevant checks, then broaden only for changed
   behavior, failures, or unresolved risks.
5. **Summarize:** distinguish implemented, exercised, and unverified behavior;
   give exact commands, results, documentation changes, and remaining limits.

Proceed autonomously on reversible choices within the user's authorization.
Clarify before changing public semantics, dependencies, supported platforms,
numerical assumptions, or compute scope that the task has not authorized.
Do not repeatedly request permission for already authorized work.

Use `gh` when available. Create an issue and branch before nontrivial coding;
update the issue with progress. If auth/network is blocked, record the reason
and next action and continue locally. Group related fixes into as few coherent
PRs as practical; keep numerical/runtime changes separate from CI policy.
PRs must explain behavior, documentation, validation, test cost,
manifest/install-profile impacts, and risks.

## Results must describe what actually ran

- Preserve requested method, actual backend, fallback reasons, and warnings
  through core results, comparisons, tools, workflows, and CLI output.
- Missing results, total backend failure, and invalid primary metrics cannot
  become successful predictions or recommendations. Undefined secondary
  metrics need explicit explanations; JSON `null` alone is not a quality flag.
- Preserve units, timestamp meaning, metric direction, train/test boundaries,
  reference provenance, and numerical tolerances. Do not change them merely
  to make a check pass. Explain authorized numerical contract changes.
- Return one authoritative CLI response after requested output operations
  finish. Keep errors and degradation visible across execution backends.
- Keep optional/heavy imports lazy. Base CLI discovery must work without
  model or agent-framework dependencies. Dependency versions live in
  `pyproject.toml` and `uv.lock`; do not add duplicate capability tables.
- Do not mutate process-global state for concurrent calls without isolation.


## Verification and maintenance cost

- Extend existing tests first. Retain tests for distinct public contracts,
  numerical oracles, and reproduced regressions; exploratory probes do not
  automatically belong in CI. Do not mirror implementation details.
- Report retained case-count and measured runtime changes. Parametrization
  reduces source duplication, not test executions. Do not prune existing
  tests solely by shared coverage or silently weaken assertions.
- Use the base-contract or affected optional profile in CONTRIBUTING.md.
  Routine verification must not implicitly download model weights, train
  substantial models, invoke paid services, or change a compute budget.
- Mocked adapters do not qualify real models, GPU behavior, or cloud backends.
  Never claim a check passed unless it actually ran.
- Keep large arrays, base64 data, and logs out of chat. Save necessary outputs
  and refer to their paths. Use conversation/PR summaries for task records;
  do not add standalone plans, reports, or ADRs unless requested.

## Documentation ownership

Update the existing source of user-facing truth with behavior changes:

| Subject | Authoritative location |
| --- | --- |
| Installation and distribution | `README.md`, `docs/distribution.qmd` |
| Contributor commands and check profiles | `CONTRIBUTING.md` |
| CLI contracts and run/job lifecycle | `docs/cli-examples.qmd` |
| Forecasting and panel limits | `docs/panel-forecasting.qmd` |
| Execution backends and resource limits | `SANDBOX.md` |
| Skills and exports | `skills/README.md`, `docs/skills-guide.qmd` |
| Evaluation evidence | `docs/evaluation.qmd`, `benchmarks/` |
| Shipped changes and future scope | `CHANGELOG.md`, `ROADMAP.md` |

Canonical skills live in `skills/`; keep the entire corresponding packaged
mirror in `ts_agents/resources/skills/` byte-identical, including supporting
files and the skills README. A PR summary does not replace these updates.

## Code Review Rules

Prioritize misleading success, numerical errors, backend substitutions,
session isolation, compatibility, and unsupported claims. Public API changes
require documentation. Check that tests assert results exist and identify
what actually ran. Flag new permanent infrastructure or test growth that lacks
a distinct maintenance benefit.

A task is complete only when the requested behavior, relevant verification,
and affected documentation are complete, or remaining blockers are stated.
