"""Darts zero-shot foundation-model adapter.

Darts, torch and huggingface_hub are imported lazily inside functions, so importing
this module stays cheap on base installs. Models are constructed with a pinned Hugging
Face revision, "fitted" once (which only loads weights; nothing is trained) and cached.
Inside :func:`model_cache_scope` (workflows and autoresearch loops) every
model stays loaded until the outermost scope exits; outside any scope at most
``_UNSCOPED_CACHE_LIMIT`` models are kept, so long-running processes do not grow.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import operator
import socket
import sys
import threading
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ..base import ForecastResult
from . import catalog
from .catalog import DEFAULT_FOUNDATION_MODEL, FoundationSpec

ACCELERATORS = ("cpu", "gpu")

_INSTALL_HINT = 'Install with: pip install "ts-agents[foundation]"'
_MODEL_CACHE: dict[tuple[str, int, int, str, int], Any] = {}
_UNSCOPED_CACHE_LIMIT = 1
_CACHE_LOCK = threading.RLock()
_SCOPE_DEPTH = 0
_SCOPE_HORIZON: Optional[int] = None
_HF_UNAVAILABLE_ERRORS = (
    "LocalEntryNotFoundError",
    "HfHubHTTPError",
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
    "OfflineModeIsEnabled",
)


class FoundationModelUnavailableError(RuntimeError):
    """Pinned foundation-model weights are neither cached nor downloadable."""


def _spec(name: str) -> FoundationSpec:
    spec = catalog.get_method(name).foundation
    if spec is None:
        raise ValueError(
            f"{name!r} is not a foundation model; choose from "
            f"{', '.join(catalog.foundation_methods())}."
        )
    return spec


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{label} must be an integer.") from None


def resolve_context_length(name: str, context_length: Optional[int], horizon: int) -> int:
    """Resolve context using the checkpoint's independent or combined window limit."""
    spec = _spec(name)
    horizon = _as_int(horizon, "horizon")
    limit = spec.max_context - horizon if spec.context_includes_horizon else spec.max_context
    if context_length is None:
        ctx = min(spec.default_context, limit)
    else:
        ctx = _as_int(context_length, "context_length")
    if ctx < 1 or ctx > limit:
        raise ValueError(
            f"{name}: context_length ({ctx}) must be between 1 and {limit} for horizon {horizon}."
        )
    return ctx


def validate_request(name: str, *, horizon: int, context_length: Optional[int] = None) -> int:
    """Validate horizon/context for ``name`` before any work; returns the resolved context."""
    spec = _spec(name)
    horizon = _as_int(horizon, "horizon")
    if horizon < 1 or horizon > spec.max_horizon:
        raise ValueError(
            f"{name}: horizon must be between 1 and {spec.max_horizon}, got {horizon}."
        )
    return resolve_context_length(name, context_length, horizon)


def _resolve_accelerator(accelerator: Optional[str]) -> str:
    resolved = accelerator or "cpu"
    if resolved not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {', '.join(ACCELERATORS)}, got {accelerator!r}.")
    return resolved


def _darts_version() -> Optional[str]:
    try:
        return metadata.version("darts")
    except metadata.PackageNotFoundError:
        return None


def model_spec(
    name: str,
    *,
    horizon: int,
    context_length: Optional[int] = None,
    accelerator: Optional[str] = "cpu",
    seed: Optional[int] = 0,
) -> dict[str, Any]:
    """Small JSON-serialisable record of how ``name`` is (re)constructed; no weights."""
    spec = _spec(name)
    ctx = validate_request(name, horizon=horizon, context_length=context_length)
    loading_horizon = _loading_horizon(int(horizon))
    record: dict[str, Any] = {
        "method": name,
        "backend": "darts",
        "darts_class": spec.darts_class,
        "hub_model_name": spec.hub_model_name,
        "hub_model_revision": spec.hub_model_revision,
        "darts_version": _darts_version(),
        "input_chunk_length": [1, ctx],
        "output_chunk_length": loading_horizon,
        "forecast_horizon": int(horizon),
        "likelihood": None,
        "accelerator": _resolve_accelerator(accelerator),
        "random_state": 0 if seed is None else int(seed),
        "zero_shot": True,
        "license": spec.license,
        "weights_saved": False,
    }
    overrides = spec.constructor_overrides(loading_horizon)
    if overrides:
        record["model_kwargs"] = overrides
    if spec.trained_horizon is not None:
        record["trained_horizon"] = spec.trained_horizon
    return record


def _install_error(name: str, missing: Sequence[str]) -> ImportError:
    return ImportError(
        f"Foundation model {name!r} needs optional packages that are not installed "
        f"({', '.join(missing)}). {_INSTALL_HINT}"
    )


def _import_model_class(spec: FoundationSpec):
    """Resolve only the requested Darts class (never iterate the catalog)."""
    try:
        import darts.models as darts_models
    except ImportError as exc:
        raise ImportError(f"Darts is not installed. {_INSTALL_HINT}") from exc
    model_cls = getattr(darts_models, spec.darts_class, None)
    try:
        from darts.utils.utils import NotImportedModule
    except ImportError:  # pragma: no cover - defensive for older darts layouts
        NotImportedModule = ()
    if model_cls is None or isinstance(model_cls, NotImportedModule) or not isinstance(model_cls, type):
        raise ImportError(
            f"darts.models.{spec.darts_class} is unavailable (torch or another backend "
            f"package is missing). {_INSTALL_HINT}"
        )
    return model_cls


def _timeseries_cls():
    from darts import TimeSeries

    return TimeSeries


def _loading_horizon(horizon: int) -> int:
    """Reserve one compatible output chunk for all horizons in a workflow."""
    with _CACHE_LOCK:
        return max(horizon, _SCOPE_HORIZON or horizon)


def _weights_unavailable(exc: BaseException) -> bool:
    """True when ``exc`` comes from a failed checkpoint lookup or download.

    Only Hugging Face Hub errors, missing files and connection failures count.
    Timeouts and interrupts (for example the autoresearch per-trial SIGALRM, a
    ``TimeoutError`` and therefore an ``OSError``) and other OS errors such as
    permission or disk-full failures propagate unchanged.
    """
    try:
        from huggingface_hub import errors as hf_errors
    except ImportError:
        hf_errors = None
    hub_types = tuple(
        error_type
        for error_type in (getattr(hf_errors, attr, None) for attr in _HF_UNAVAILABLE_ERRORS)
        if isinstance(error_type, type)
    )
    lookup_types = hub_types + (FileNotFoundError, ConnectionError, socket.gaierror)
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, InterruptedError)):
            return False
        if isinstance(current, lookup_types):
            return True
        current = current.__cause__ or current.__context__
    return False


def _unavailable_message(spec: FoundationSpec) -> str:
    # Must not look like a missing-package failure to forecast._is_dependency_failure.
    return (
        f"weights for {spec.hub_model_name}@{spec.hub_model_revision} are not cached and "
        "could not be downloaded; run once with network access, pre-populate HF_HOME, "
        "or allow sandbox network (--allow-network)"
    )


def _constructor_kwargs(
    spec: FoundationSpec, *, ctx: int, horizon: int, accelerator: str, seed: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_chunk_length": (1, ctx),
        "output_chunk_length": horizon,
        "hub_model_name": spec.hub_model_name,
        "hub_model_revision": spec.hub_model_revision,
        "random_state": seed,
        "save_checkpoints": False,
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": 1,
            "enable_progress_bar": False,
            "logger": False,
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    }
    kwargs.update(spec.constructor_overrides(horizon))
    return kwargs


def _load(name: str, ctx: int, horizon: int, accelerator: str, seed: int, fit_series: Any):
    """Construct + fit (weight load only) once per cache key."""
    horizon = _loading_horizon(horizon)
    validate_request(name, horizon=horizon, context_length=ctx)
    key = (name, ctx, horizon, accelerator, seed)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.pop(key, None)
        if cached is not None:
            # Re-insert so the most recently used model is evicted last.
            _MODEL_CACHE[key] = cached
            return cached
    spec = _spec(name)
    model_cls = _import_model_class(spec)
    try:
        # Trainer options disable progress/loggers; diagnostics use stderr.
        # Never change global stdout, logger levels or warning filters.
        model = model_cls(
            **_constructor_kwargs(spec, ctx=ctx, horizon=horizon, accelerator=accelerator, seed=seed)
        )
        model.fit(fit_series)
    except Exception as exc:
        if _weights_unavailable(exc):
            raise FoundationModelUnavailableError(_unavailable_message(spec)) from exc
        raise
    evicted = []
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = model
        if _SCOPE_DEPTH == 0:
            while len(_MODEL_CACHE) > _UNSCOPED_CACHE_LIMIT:
                evicted.append(_MODEL_CACHE.pop(next(iter(_MODEL_CACHE))))
    if evicted:
        del evicted
        _release_memory()
    return model


def _release_memory() -> None:
    """Free dropped models now.

    Darts/Lightning models sit in reference cycles (model <-> trainer), so dropping the
    cache entry alone leaves the weights resident until a later GC pass; measured at
    roughly 0.9 GB per TimesFM 2.5 model. Collect explicitly and release cached GPU
    blocks when torch is already loaded.
    """
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort only
            pass


def clear_model_cache() -> None:
    """Drop every cached model and free its memory."""
    with _CACHE_LOCK:
        had_models = bool(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    if had_models:
        _release_memory()


@contextmanager
def model_cache_scope(*, horizon: Optional[int] = None) -> Iterator[None]:
    """Keep scoped models, then release them; optionally reserve a shared horizon.

    Forecast workflows know both validation and future horizons before loading.
    Reserving their maximum loads one checkpoint with one output chunk rather than
    keeping two copies. A reentrant lock serializes scopes and model inference so
    concurrent threads cannot reuse or release a model during another prediction.
    """
    global _SCOPE_DEPTH, _SCOPE_HORIZON
    with _CACHE_LOCK:
        previous_horizon = _SCOPE_HORIZON
        _SCOPE_HORIZON = max(horizon or 0, previous_horizon or 0) or None
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
            _SCOPE_HORIZON = previous_horizon
            if _SCOPE_DEPTH == 0:
                clear_model_cache()


def forecast_arrays(
    arrays: Sequence[Any],
    *,
    model: str,
    horizon: int,
    context_length: Optional[int] = None,
    accelerator: Optional[str] = "cpu",
    seed: Optional[int] = 0,
) -> list[np.ndarray]:
    """Zero-shot point forecasts (float64) for each 1-D history in ``arrays``."""
    ctx = validate_request(model, horizon=horizon, context_length=context_length)
    horizon = int(horizon)
    accelerator = _resolve_accelerator(accelerator)
    seed = 0 if seed is None else _as_int(seed, "seed")
    missing = catalog.missing_modules(model)
    if missing:
        raise _install_error(model, missing)
    if len(arrays) == 0:
        raise ValueError("Foundation-model forecasts need at least one series.")
    contexts: list[np.ndarray] = []
    for position, values in enumerate(arrays):
        history = np.asarray(values, dtype=np.float64)
        if history.ndim != 1:
            raise ValueError(f"Series {position} must be one-dimensional.")
        if history.size == 0:
            raise ValueError(f"Series {position} is empty.")
        if not np.isfinite(history).all():
            raise ValueError(f"Series {position} contains missing or non-finite values.")
        # TimesFM/PatchTST-FM reject float64 tensors; Chronos-2 casts internally.
        context = history[-ctx:]
        if np.any(np.abs(context) > np.finfo(np.float32).max):
            raise ValueError(f"Series {position} context exceeds the finite float32 range.")
        contexts.append(context.astype(np.float32))

    time_series_cls = _timeseries_cls()
    series = [
        time_series_cls.from_times_and_values(pd.RangeIndex(len(values)), values, columns=["y"])
        for values in contexts
    ]
    longest = max(range(len(series)), key=lambda index: len(contexts[index]))
    with _CACHE_LOCK:
        fitted = _load(model, ctx, horizon, accelerator, seed, series[longest])
        predictions = fitted.predict(n=horizon, series=series, verbose=False)
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    if len(predictions) != len(series):
        raise RuntimeError(f"{model} returned {len(predictions)} forecasts for {len(series)} series.")
    outputs: list[np.ndarray] = []
    for prediction in predictions:
        values = np.asarray(prediction.values(copy=False), dtype=np.float64).reshape(-1)
        if values.shape != (horizon,) or not np.isfinite(values).all():
            raise RuntimeError(f"{model} returned a malformed or non-finite forecast.")
        outputs.append(values)
    return outputs


def forecast_panel_frame(
    history: pd.DataFrame,
    *,
    model: str,
    freq: str,
    horizon: int,
    context_length: Optional[int] = None,
    accelerator: Optional[str] = "cpu",
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Forecast every ``unique_id`` of a long panel; returns ``unique_id, ds, <model>``."""
    if history.empty:
        raise ValueError("Panel history is empty.")
    ids: list[Any] = []
    ends: list[pd.Timestamp] = []
    arrays: list[np.ndarray] = []
    for series_id, group in history.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        ids.append(series_id)
        ends.append(group.ds.iloc[-1])
        arrays.append(group.y.to_numpy(dtype=np.float64))
    forecasts = forecast_arrays(
        arrays,
        model=model,
        horizon=horizon,
        context_length=context_length,
        accelerator=accelerator,
        seed=seed,
    )
    frames = []
    for series_id, end, values in zip(ids, ends, forecasts):
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": series_id,
                    "ds": pd.date_range(end, periods=int(horizon) + 1, freq=freq)[1:],
                    model: values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def forecast_foundation(
    series: Any,
    *,
    model: str = DEFAULT_FOUNDATION_MODEL,
    horizon: int = 10,
    season_length: Optional[int] = None,
    context_length: Optional[int] = None,
    accelerator: Optional[str] = "cpu",
    seed: Optional[int] = 0,
    **_: Any,
) -> ForecastResult:
    """Zero-shot point forecast of one series; ``season_length`` is accepted and ignored."""
    forecast = forecast_arrays(
        [series],
        model=model,
        horizon=horizon,
        context_length=context_length,
        accelerator=accelerator,
        seed=seed,
    )[0]
    return ForecastResult(method=model, forecast=forecast, horizon=int(horizon))
