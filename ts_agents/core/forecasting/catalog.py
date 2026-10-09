"""Stdlib-only forecasting method catalog shared by every CLI, tool and agent surface.

Importing this module never imports numpy, pandas, statsforecast, Nixtla backends or
Darts; availability is probed with ``importlib.util.find_spec`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
import sys
from typing import Any, Callable, Iterable, Mapping, Optional

FOUNDATION_EXTRA = "foundation"
FOUNDATION_MODULES = ("darts", "torch", "huggingface_hub")
MODULE_DISTRIBUTIONS = {
    "sklearn": "scikit-learn",
    "darts": "darts",
    "torch": "torch",
    "huggingface_hub": "huggingface-hub",
}
FOUNDATION_DISTRIBUTIONS = {module: MODULE_DISTRIBUTIONS[module] for module in FOUNDATION_MODULES}
DEFAULT_FOUNDATION_MODEL = "chronos2_small"
SURFACES = ("series", "panel", "autoresearch")

_STATISTICAL_OPTIONS = frozenset({"season_length", "level"})
_FOUNDATION_OPTIONS = frozenset({"context_length", "accelerator", "seed"})
_FOUNDATION_SURFACES = ("series", "panel", "autoresearch")


@dataclass(frozen=True, kw_only=True)
class FoundationSpec:
    """Pinned Darts checkpoint for one zero-shot foundation model."""

    darts_class: str
    hub_model_name: str
    hub_model_revision: str
    max_horizon: int
    max_context: int
    context_includes_horizon: bool = False
    default_context: int = 512
    trained_horizon: Optional[int] = None
    # Extra constructor kwargs applied only when horizon > native_max_horizon.
    long_horizon_kwargs: Optional[Mapping[str, Any]] = field(default=None, hash=False)
    native_max_horizon: Optional[int] = None
    license: str
    approx_weights_mb: int
    family_label: str

    def constructor_overrides(self, horizon: int) -> dict[str, Any]:
        """Return the horizon-dependent constructor kwargs (empty for most models)."""
        if (
            self.long_horizon_kwargs
            and self.native_max_horizon is not None
            and horizon > self.native_max_horizon
        ):
            return dict(self.long_horizon_kwargs)
        return {}


@dataclass(frozen=True, kw_only=True)
class MethodSpec:
    """One forecasting method and the surfaces, extra and modules it needs."""

    name: str
    family: str
    label: str
    modules: tuple[str, ...]
    extra: Optional[str]
    surfaces: tuple[str, ...]
    accepted_options: frozenset[str]
    backend: str = "native"
    zero_shot: bool = False
    foundation: Optional[FoundationSpec] = None


def _foundation_method(name: str, label: str, spec: FoundationSpec) -> MethodSpec:
    return MethodSpec(
        name=name,
        family="foundation",
        label=label,
        modules=FOUNDATION_MODULES,
        extra=FOUNDATION_EXTRA,
        surfaces=_FOUNDATION_SURFACES,
        accepted_options=_FOUNDATION_OPTIONS,
        backend="darts",
        zero_shot=True,
        foundation=spec,
    )


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="seasonal_naive",
        family="baseline",
        label="Seasonal naive",
        modules=(),
        extra=None,
        surfaces=("series", "panel"),
        accepted_options=_STATISTICAL_OPTIONS,
    ),
    MethodSpec(
        name="arima",
        family="statistical",
        label="AutoARIMA",
        modules=("statsforecast",),
        extra="forecasting",
        surfaces=("series",),
        accepted_options=_STATISTICAL_OPTIONS,
    ),
    MethodSpec(
        name="ets",
        family="statistical",
        label="AutoETS",
        modules=("statsforecast",),
        extra="forecasting",
        surfaces=("series",),
        accepted_options=_STATISTICAL_OPTIONS,
    ),
    MethodSpec(
        name="theta",
        family="statistical",
        label="AutoTheta",
        modules=("statsforecast",),
        extra="forecasting",
        surfaces=("series",),
        accepted_options=_STATISTICAL_OPTIONS,
    ),
    MethodSpec(
        name="lightgbm",
        family="ml",
        label="MLForecast LightGBM",
        modules=("mlforecast", "lightgbm"),
        extra="ml",
        surfaces=("panel",),
        accepted_options=frozenset(),
    ),
    MethodSpec(
        name="histgbm",
        family="ml",
        label="MLForecast HistGradientBoosting",
        modules=("mlforecast", "sklearn"),
        extra="ml",
        surfaces=("panel",),
        accepted_options=frozenset(),
    ),
    MethodSpec(
        name="nhits",
        family="neural",
        label="NeuralForecast NHITS",
        modules=("neuralforecast", "torch"),
        extra="neural",
        surfaces=("panel",),
        accepted_options=frozenset(),
    ),
    _foundation_method(
        "chronos2_small",
        "Chronos-2 small (Darts, zero-shot)",
        FoundationSpec(
            darts_class="Chronos2Model",
            hub_model_name="autogluon/chronos-2-small",
            hub_model_revision="ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
            max_horizon=1024,
            max_context=8192,
            license="apache-2.0",
            approx_weights_mb=112,
            family_label="Chronos-2",
        ),
    ),
    _foundation_method(
        "chronos2",
        "Chronos-2 (Darts, zero-shot)",
        FoundationSpec(
            darts_class="Chronos2Model",
            hub_model_name="amazon/chronos-2",
            hub_model_revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
            max_horizon=1024,
            max_context=8192,
            license="apache-2.0",
            approx_weights_mb=478,
            family_label="Chronos-2",
        ),
    ),
    _foundation_method(
        "timesfm2p5",
        "TimesFM 2.5 200M (Darts, zero-shot)",
        FoundationSpec(
            darts_class="TimesFM2p5Model",
            hub_model_name="google/timesfm-2.5-200m-pytorch",
            hub_model_revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            max_horizon=1024,
            max_context=16384,
            context_includes_horizon=True,
            long_horizon_kwargs={"use_longer_projection_head": True},
            native_max_horizon=128,
            license="apache-2.0",
            approx_weights_mb=925,
            family_label="TimesFM",
        ),
    ),
    _foundation_method(
        "patchtst_fm",
        "PatchTST-FM r1 (Darts, zero-shot)",
        FoundationSpec(
            darts_class="PatchTSTFMModel",
            hub_model_name="ibm-granite/granite-timeseries-patchtst-fm-r1",
            hub_model_revision="151f9c6d576281b95c2ff784d0863bd3f12c80f1",
            max_horizon=1024,
            max_context=8192,
            context_includes_horizon=True,
            trained_horizon=64,
            license="apache-2.0",
            approx_weights_mb=1032,
            family_label="PatchTST-FM",
        ),
    ),
)

_BY_NAME: dict[str, MethodSpec] = {method.name: method for method in METHODS}


def _check_surface(surface: str) -> None:
    if surface not in SURFACES:
        raise ValueError(f"Unknown method surface {surface!r}; expected one of {', '.join(SURFACES)}.")


def methods_for(surface: str) -> list[str]:
    """Method names usable on ``surface`` (series, panel or autoresearch), in catalog order."""
    _check_surface(surface)
    return [method.name for method in METHODS if surface in method.surfaces]


def get_method(name: str) -> MethodSpec:
    """Return the catalog entry for ``name``; unknown names raise ``ValueError``."""
    try:
        return _BY_NAME[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown forecasting method {name!r}; choose from {', '.join(_BY_NAME)}."
        ) from None


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # find_spec raises ValueError for modules injected without a __spec__.
        return module in sys.modules


def missing_modules(name: str) -> list[str]:
    """Import names required by ``name`` that are not installed (find_spec only)."""
    return [module for module in get_method(name).modules if not _module_present(module)]


def is_available(name: str) -> bool:
    return not missing_modules(name)


def _as_names(methods: Iterable[str] | str | None) -> list[str]:
    if methods is None:
        return []
    if isinstance(methods, str):
        return [item.strip() for item in methods.split(",") if item.strip()]
    return [str(item).strip() for item in methods if str(item).strip()]


def required_extras_for(methods: Iterable[str] | str | None) -> list[str]:
    """Sorted unique extras needed by ``methods``; unknown names are left to validation."""
    extras: set[str] = set()
    for name in _as_names(methods):
        extra = _BY_NAME[name].extra if name in _BY_NAME else None
        if extra:
            extras.add(extra)
    return sorted(extras)


def install_hint_for(methods: Iterable[str] | str | None) -> Optional[str]:
    """Install hint covering only the requested methods whose modules are missing."""
    missing = [name for name in _as_names(methods) if name in _BY_NAME and missing_modules(name)]
    extras = required_extras_for(missing)
    if not extras:
        return None
    return f"Install `ts-agents[{','.join(extras)}]`."


def method_extras(surface: str) -> dict[str, Optional[str]]:
    """Map each method on ``surface`` to its extra (None for base methods)."""
    return {name: _BY_NAME[name].extra for name in methods_for(surface)}


def foundation_methods(surface: Optional[str] = None) -> list[str]:
    """Foundation-model keys, optionally limited to one surface."""
    names = methods_for(surface) if surface is not None else list(_BY_NAME)
    return [name for name in names if _BY_NAME[name].foundation is not None]


def foundation_checkpoints(surface: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """JSON-friendly pinned checkpoint metadata per foundation model."""
    checkpoints: dict[str, dict[str, Any]] = {}
    for name in foundation_methods(surface):
        spec = _BY_NAME[name].foundation
        assert spec is not None
        checkpoints[name] = {
            "darts_class": spec.darts_class,
            "hub_model_name": spec.hub_model_name,
            "hub_model_revision": spec.hub_model_revision,
            "max_horizon": spec.max_horizon,
            "max_context": spec.max_context,
            "default_context": spec.default_context,
            "trained_horizon": spec.trained_horizon,
            "license": spec.license,
            "approx_weights_mb": spec.approx_weights_mb,
            "family": spec.family_label,
        }
    return checkpoints


def get_series_forecaster(name: str) -> Callable[..., Any]:
    """Return ``f(series, horizon=..., **options) -> ForecastResult`` for a series method.

    Known shared options irrelevant to this method are ignored. Unknown option names
    raise TypeError before fitting, so spelling mistakes cannot change an evaluation.
    """
    spec = get_method(name)
    if "series" not in spec.surfaces:
        raise ValueError(
            f"{name!r} is a panel-only method; series methods are: "
            f"{', '.join(methods_for('series'))}."
        )
    if spec.foundation is not None:

        def target(series, *, horizon, **options):
            from .foundation import forecast_foundation

            return forecast_foundation(series, model=name, horizon=horizon, **options)
    else:

        def target(series, *, horizon, **options):
            statistical = importlib.import_module(f"{__package__}.statistical")
            return getattr(statistical, f"forecast_{name}")(series, horizon=horizon, **options)

    accepted = spec.accepted_options

    def forecaster(series, horizon: int = 10, **options):
        unknown = set(options) - (_STATISTICAL_OPTIONS | _FOUNDATION_OPTIONS)
        if unknown:
            raise TypeError(f"Unknown forecasting options: {', '.join(sorted(unknown))}")
        kept = {key: value for key, value in options.items() if key in accepted}
        return target(series, horizon=horizon, **kept)

    forecaster.__name__ = f"forecast_{name}"
    forecaster.__qualname__ = forecaster.__name__
    return forecaster
