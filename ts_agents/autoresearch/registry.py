"""Registry for built-in autoresearch loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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


FOUNDATION_CHRONOS_LOOP_NAME = "foundation-chronos-smoke"
FOUNDATION_CHRONOS_MODEL = "amazon/chronos-t5-tiny"
FOUNDATION_CHRONOS_TASK = "foundation-model-smoke"
FOUNDATION_CHRONOS_HORIZON = 18
FOUNDATION_CHRONOS_SEASON_LENGTH = 12
FOUNDATION_CHRONOS_DEFAULT_SERIES = ["M4"]
FOUNDATION_CHRONOS_MODEL_SCOPE = "single_chronos_zero_shot_smoke"
FOUNDATION_CHRONOS_MODEL_SCOPE_LABEL = "single Chronos-family zero-shot smoke path"
FOUNDATION_CHRONOS_INSTALL_HINT = "pip install 'ts-agents[foundation]'"
FOUNDATION_CHRONOS_EXTERNAL_CONTEXT = "benchmarks/external/gift_eval_snapshot.json"


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
    FOUNDATION_CHRONOS_LOOP_NAME: AutoresearchLoopDefinition(
        name=FOUNDATION_CHRONOS_LOOP_NAME,
        task=FOUNDATION_CHRONOS_TASK,
        description=(
            "Run a scoped Chronos zero-shot forecasting smoke check on the vendored "
            "M4 Monthly mini panel. This is an executable TSFM path, not a model hub."
        ),
        dataset="data/m4_monthly_mini.csv",
        models=[FOUNDATION_CHRONOS_MODEL],
        primary_metric="smape",
        secondary_metrics=["mae", "rmse", "elapsed_seconds"],
        budget=AutoresearchBudget(
            timeout_seconds=30 * 60,
            vcpu=4,
            memory_mb=16 * 1024,
            disk_mb=20 * 1024,
            max_trials=1,
            notes=[
                "Executes only a single Chronos-family zero-shot path by default.",
                "Dry runs do not require heavy foundation-model dependencies.",
                "Executable runs lazy-import chronos-forecasting and torch.",
            ],
        ),
        required_extras=["foundation"],
        dependency_rules=(
            AutoresearchDependencyRule(
                modules=(("chronos", "chronos-forecasting"), ("torch", "torch")),
                install_extra="foundation",
                skip_on_dry_run=True,
                label="foundation-model dependencies",
            ),
        ),
        capabilities={
            "status": "executable_optional_dependency",
            "generates_metrics": True,
            "model_scope": FOUNDATION_CHRONOS_MODEL_SCOPE,
            "model_scope_label": FOUNDATION_CHRONOS_MODEL_SCOPE_LABEL,
            "install_hint": FOUNDATION_CHRONOS_INSTALL_HINT,
            "horizon": FOUNDATION_CHRONOS_HORIZON,
            "season_length": FOUNDATION_CHRONOS_SEASON_LENGTH,
            "default_series": FOUNDATION_CHRONOS_DEFAULT_SERIES,
            "max_trials_semantics": "number of Chronos zero-shot forecast rows",
            "search_style": "smoke check",
            "ranking_rule": "lowest sMAPE on the M4 mini holdout; MAE and RMSE are tie-breakers",
            "external_benchmark_context": FOUNDATION_CHRONOS_EXTERNAL_CONTEXT,
        },
    ),
    "foundation-gpu-plan": AutoresearchLoopDefinition(
        name="foundation-gpu-plan",
        task="foundation-model-plan",
        description=(
            "Materialize an RTX PRO 6000 Blackwell-oriented, plan-only foundation-model "
            "research recipe covering Chronos-family forecasting and MOMENT classification."
        ),
        dataset="vendored M4 mini and generated/vendored labeled streams for smoke runs",
        models=["amazon/chronos-2", "amazon/chronos-t5-small", "AutonLab/MOMENT-1-large"],
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
            ],
        ),
        required_extras=[],
        capabilities={
            "status": "plan-only",
            "generates_metrics": False,
            "forecasting_model": "amazon/chronos-2",
            "forecasting_model_revision": "254b5357164a84326913b0695216f690752ac55d",
            "forecasting_finetune_fallback": "amazon/chronos-t5-small",
            "forecasting_finetune_revision": "4753ebecb99f65f84cd2823c56f7ab22b02ac303",
            "classification_model": "AutonLab/MOMENT-1-large",
            "classification_model_revision": "3582f9d7f033eea9d43e6a802ba0e36d5f26b57c",
            "smoke_budget": "30 minutes, 1 seed, 1 epoch or 1000 max steps",
            "full_budget": "4 hours, 3 seeds, early stopping, bf16, checkpoint cap 5 GiB",
        },
    ),
}


def list_loops() -> list[AutoresearchLoopDefinition]:
    """Return built-in autoresearch loop definitions."""
    return list(_LOOPS.values())


def get_loop(name: str) -> AutoresearchLoopDefinition:
    """Return a built-in autoresearch loop definition by name."""
    try:
        return _LOOPS[name]
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
