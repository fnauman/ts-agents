"""Registry for built-in autoresearch loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ts_agents.core.forecasting.catalog import (
    DEFAULT_FOUNDATION_MODEL,
    FOUNDATION_DISTRIBUTIONS,
    FOUNDATION_EXTRA,
    FOUNDATION_MODULES,
    foundation_checkpoints,
    get_method,
    methods_for,
)


@dataclass(frozen=True)
class AutoresearchBudget:
    """Default resource budget for an autoresearch loop."""

    timeout_seconds: int
    vcpu: int
    memory_mb: int
    disk_mb: int
    max_trials: int
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AutoresearchDependencyRule:
    """Optional-dependency requirement for running a loop on the host.

    ``modules`` maps import names to pip distribution names. When ``models``
    is set the rule only applies if the selected models intersect it, and
    ``skip_on_dry_run`` exempts dry runs that never import the dependency.
    """

    modules: tuple[tuple[str, str], ...]
    install_extra: str
    models: tuple[str, ...] | None = None
    skip_on_dry_run: bool = False
    label: str = "dependencies"


@dataclass(frozen=True)
class AutoresearchLoopDefinition:
    """Machine-readable metadata for one autoresearch loop."""

    name: str
    task: str
    description: str
    dataset: str
    models: list[str]
    primary_metric: str
    secondary_metrics: list[str]
    budget: AutoresearchBudget
    required_extras: list[str] = field(default_factory=list)
    output_root: str = "outputs/autoresearch"
    capabilities: dict[str, Any] = field(default_factory=dict)
    dependency_rules: tuple[AutoresearchDependencyRule, ...] = ()
    # Models run when --models is omitted; empty means every entry in ``models``.
    default_models: list[str] = field(default_factory=list)


FOUNDATION_SMOKE_LOOP_NAME = "foundation-smoke"
FOUNDATION_SMOKE_TASK = "foundation-model-smoke"
FOUNDATION_SMOKE_HORIZON = 18
FOUNDATION_SMOKE_SEASON_LENGTH = 12
FOUNDATION_SMOKE_CONTEXT_LENGTH = 512
FOUNDATION_SMOKE_DEFAULT_SERIES = ["M4"]
FOUNDATION_SMOKE_DEFAULT_MODELS = [DEFAULT_FOUNDATION_MODEL]
FOUNDATION_SMOKE_MODEL_SCOPE = "darts_foundation_zero_shot_smoke"
FOUNDATION_SMOKE_MODEL_SCOPE_LABEL = (
    "Darts zero-shot foundation-model smoke path (selected checkpoints only)"
)
FOUNDATION_SMOKE_INSTALL_HINT = f"pip install 'ts-agents[{FOUNDATION_EXTRA}]'"

FOUNDATION_GPU_PLAN_FORECAST_MODEL = "chronos2"
FOUNDATION_GPU_PLAN_COMPARATORS = ["timesfm2p5", "patchtst_fm"]
_GPU_PLAN_FORECAST_SPEC = get_method(FOUNDATION_GPU_PLAN_FORECAST_MODEL).foundation
MOMENT_MODEL = "AutonLab/MOMENT-1-large"
MOMENT_REVISION = "3582f9d7f033eea9d43e6a802ba0e36d5f26b57c"
MOMENT_SUPPORT = "external_plan_only"
MOMENT_SUPPORT_NOTE = "requires momentfm, not installed by any ts-agents extra"

# Deprecated loop names kept resolvable for one release; never listed.
_LOOP_ALIASES: dict[str, str] = {
    "foundation-chronos-smoke": FOUNDATION_SMOKE_LOOP_NAME,
}

_DAYTONA_BUDGET = AutoresearchBudget(
    timeout_seconds=20 * 60,
    vcpu=4,
    memory_mb=8 * 1024,
    disk_mb=10 * 1024,
    max_trials=60,
    notes=[
        "Designed for Daytona sandboxes capped at 4 vCPU, 8 GiB RAM, and 10 GiB disk.",
        "Default data sources are vendored or generated locally; no dataset download is required.",
    ],
)


_LOOPS: dict[str, AutoresearchLoopDefinition] = {
    "forecast-daytona": AutoresearchLoopDefinition(
        name="forecast-daytona",
        task="forecasting",
        description=(
            "Compare statistical forecasting baselines on the vendored M4 Monthly mini panel "
            "under constrained Daytona-style resources."
        ),
        dataset="data/m4_monthly_mini.csv",
        models=["seasonal_naive", "theta", "ets", "arima"],
        primary_metric="smape",
        secondary_metrics=["mae", "rmse", "mape", "elapsed_seconds", "failure_rate"],
        budget=_DAYTONA_BUDGET,
        required_extras=["forecasting"],
        dependency_rules=(
            AutoresearchDependencyRule(
                modules=(("statsforecast", "statsforecast"),),
                install_extra="forecasting",
                models=("theta", "ets", "arima"),
            ),
        ),
        capabilities={
            "horizon": 18,
            "season_length": 12,
            "default_series": ["M4", "M10", "M100", "M1000", "M1002"],
            "rolling_origins": 2,
            "max_trials_semantics": "number of model/evaluation-spec rows",
            "search_style": "benchmark sweep",
            "ranking_rule": "lowest mean sMAPE across holdout and rolling-origin rows; MAE and RMSE are tie-breakers",
        },
    ),
    "classify-daytona": AutoresearchLoopDefinition(
        name="classify-daytona",
        task="classification",
        description=(
            "Compare windowed time-series classifiers on a vendored or generated labeled "
            "activity stream under constrained Daytona-style resources."
        ),
        dataset="data/wisdm_subset.csv with generated synthetic fallback",
        models=["knn", "minirocket", "rocket"],
        primary_metric="balanced_accuracy",
        secondary_metrics=["best_window_size", "n_windows", "elapsed_seconds", "failure_rate"],
        budget=AutoresearchBudget(
            timeout_seconds=20 * 60,
            vcpu=4,
            memory_mb=8 * 1024,
            disk_mb=10 * 1024,
            max_trials=3,
            notes=list(_DAYTONA_BUDGET.notes),
        ),
        required_extras=["classification"],
        dependency_rules=(
            AutoresearchDependencyRule(
                modules=(("aeon", "aeon"), ("sklearn", "scikit-learn")),
                install_extra="classification",
                models=("knn", "minirocket", "rocket"),
            ),
        ),
        capabilities={
            "window_sizes": [32, 64, 96, 128, 160],
            "labeling": "strict",
            "balance": "segment_cap",
            "max_windows_per_segment": 25,
            "n_splits": 3,
            "seed": 1337,
            "max_trials_semantics": "number of model/evaluation-spec rows",
            "search_style": "benchmark sweep with per-classifier window-size selection",
            "ranking_rule": "highest balanced accuracy from window-selection CV; smaller selected window and more retained windows are tie-breakers",
        },
    ),
    FOUNDATION_SMOKE_LOOP_NAME: AutoresearchLoopDefinition(
        name=FOUNDATION_SMOKE_LOOP_NAME,
        task=FOUNDATION_SMOKE_TASK,
        description=(
            "Run a scoped Darts zero-shot foundation-model forecasting smoke check "
            "(Chronos-2, TimesFM 2.5, PatchTST-FM) on the vendored M4 Monthly mini panel."
        ),
        dataset="data/m4_monthly_mini.csv",
        models=methods_for("autoresearch"),
        default_models=list(FOUNDATION_SMOKE_DEFAULT_MODELS),
        primary_metric="smape",
        secondary_metrics=["mae", "rmse", "elapsed_seconds"],
        budget=AutoresearchBudget(
            timeout_seconds=30 * 60,
            vcpu=4,
            memory_mb=16 * 1024,
            disk_mb=20 * 1024,
            max_trials=4,
            notes=[
                f"Runs only {DEFAULT_FOUNDATION_MODEL} unless --models selects other checkpoints.",
                "Only the selected checkpoints download into HF_HOME; nothing is trained.",
                "Dry runs do not require heavy foundation-model dependencies.",
                "Executable runs lazy-import darts, torch and huggingface_hub.",
            ],
        ),
        required_extras=[FOUNDATION_EXTRA],
        dependency_rules=(
            AutoresearchDependencyRule(
                modules=tuple(
                    (module, FOUNDATION_DISTRIBUTIONS[module])
                    for module in FOUNDATION_MODULES
                ),
                install_extra=FOUNDATION_EXTRA,
                skip_on_dry_run=True,
                label="foundation-model dependencies",
            ),
        ),
        capabilities={
            "status": "executable_optional_dependency",
            "generates_metrics": True,
            "model_scope": FOUNDATION_SMOKE_MODEL_SCOPE,
            "model_scope_label": FOUNDATION_SMOKE_MODEL_SCOPE_LABEL,
            "install_hint": FOUNDATION_SMOKE_INSTALL_HINT,
            "backend": "darts",
            "execution_mode": "zero_shot",
            "horizon": FOUNDATION_SMOKE_HORIZON,
            "season_length": FOUNDATION_SMOKE_SEASON_LENGTH,
            "context_length": FOUNDATION_SMOKE_CONTEXT_LENGTH,
            "default_series": FOUNDATION_SMOKE_DEFAULT_SERIES,
            "default_models": list(FOUNDATION_SMOKE_DEFAULT_MODELS),
            "checkpoints": foundation_checkpoints("autoresearch"),
            "deprecated_aliases": sorted(_LOOP_ALIASES),
            "max_trials_semantics": "number of zero-shot foundation-model forecast rows",
            "search_style": "smoke check",
            "ranking_rule": "lowest sMAPE on the M4 mini holdout; MAE and RMSE are tie-breakers",
        },
    ),
    "foundation-gpu-plan": AutoresearchLoopDefinition(
        name="foundation-gpu-plan",
        task="foundation-model-plan",
        description=(
            "Materialize an RTX PRO 6000 Blackwell-oriented, plan-only foundation-model "
            "research recipe covering Darts Chronos-2 fine-tuning, TimesFM 2.5 and "
            "PatchTST-FM zero-shot comparators, and external MOMENT classification."
        ),
        dataset="vendored M4 mini and generated/vendored labeled streams for smoke runs",
        models=[
            FOUNDATION_GPU_PLAN_FORECAST_MODEL,
            *FOUNDATION_GPU_PLAN_COMPARATORS,
            MOMENT_MODEL,
        ],
        primary_metric="not_applicable_plan_only",
        secondary_metrics=["planned_wall_time", "planned_gpu_memory_gb", "checkpoint_cap_gb"],
        budget=AutoresearchBudget(
            timeout_seconds=4 * 60 * 60,
            vcpu=16,
            memory_mb=64 * 1024,
            disk_mb=200 * 1024,
            max_trials=6,
            notes=[
                "Assumes one RTX PRO 6000 Blackwell GPU.",
                "Heavy foundation-model packages are intentionally optional and lazy-loaded.",
                f"{MOMENT_MODEL} is {MOMENT_SUPPORT}: {MOMENT_SUPPORT_NOTE}.",
            ],
        ),
        required_extras=[],
        capabilities={
            "status": "plan-only",
            "generates_metrics": False,
            "forecasting_backend": "darts",
            "forecasting_method": FOUNDATION_GPU_PLAN_FORECAST_MODEL,
            "forecasting_darts_class": _GPU_PLAN_FORECAST_SPEC.darts_class,
            "forecasting_model": _GPU_PLAN_FORECAST_SPEC.hub_model_name,
            "forecasting_model_revision": _GPU_PLAN_FORECAST_SPEC.hub_model_revision,
            "forecasting_finetune": "Chronos2Model(enable_finetuning=True)",
            "zero_shot_comparators": list(FOUNDATION_GPU_PLAN_COMPARATORS),
            "classification_model": MOMENT_MODEL,
            "classification_model_revision": MOMENT_REVISION,
            "classification_support": f"{MOMENT_SUPPORT}: {MOMENT_SUPPORT_NOTE}",
            "smoke_budget": "30 minutes, 1 seed, 1 epoch or 1000 max steps",
            "full_budget": "4 hours, 3 seeds, early stopping, bf16, checkpoint cap 5 GiB",
        },
    ),
}


def list_loops() -> list[AutoresearchLoopDefinition]:
    """Return built-in autoresearch loop definitions (deprecated aliases excluded)."""
    return list(_LOOPS.values())


def canonical_loop_name(name: str) -> str:
    """Resolve a deprecated loop alias to its current name; other names pass through."""
    return _LOOP_ALIASES.get(name, name)


def deprecated_alias_warning(name: str) -> str | None:
    """Return a deprecation warning when ``name`` is an alias, else None."""
    canonical = _LOOP_ALIASES.get(name)
    if canonical is None:
        return None
    return (
        f"Autoresearch loop '{name}' is deprecated and will be removed in a future "
        f"release; use '{canonical}' instead."
    )


def get_loop(name: str) -> AutoresearchLoopDefinition:
    """Return a built-in autoresearch loop definition by name or deprecated alias."""
    try:
        return _LOOPS[canonical_loop_name(name)]
    except KeyError as exc:
        available = ", ".join(sorted(_LOOPS))
        raise KeyError(f"Unknown autoresearch loop '{name}'. Available: {available}.") from exc


def loop_to_dict(loop: AutoresearchLoopDefinition) -> dict[str, Any]:
    """Convert a loop definition to a stable JSON-compatible dictionary."""
    payload = asdict(loop)
    payload["cli_templates"] = [
        f"ts-agents autoresearch show {loop.name} --json",
        f"ts-agents autoresearch run {loop.name} --json",
    ]
    return payload
