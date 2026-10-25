"""Tool-layer adapters that are not plain core functions.

Most tools call core analysis functions directly. The adapters here cover
foundation-model forecasting (lazy Darts loading, bounded model cache) and
file-backed panel forecasting that dispatches the forecast-panel workflow.
"""

import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional
import uuid

import numpy as np

from ts_agents.contracts import ToolPayload
from ts_agents.core.forecasting.catalog import DEFAULT_FOUNDATION_MODEL


_FALLBACK_ARTIFACT_DIR: Optional[Path] = None
_TOOL_ARTIFACT_DIR_ENV = "TS_AGENTS_TOOL_ARTIFACT_DIR"


def _get_artifact_dir() -> Path:
    global _FALLBACK_ARTIFACT_DIR

    env_dir = os.environ.get(_TOOL_ARTIFACT_DIR_ENV)
    if env_dir:
        artifact_dir = Path(env_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    if _FALLBACK_ARTIFACT_DIR is None:
        _FALLBACK_ARTIFACT_DIR = Path(
            tempfile.mkdtemp(prefix="ts_agents_tool_artifacts_")
        )
    _FALLBACK_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return _FALLBACK_ARTIFACT_DIR


def _foundation_model_name(model: str) -> str:
    from ts_agents.core.forecasting.catalog import foundation_methods

    names = foundation_methods("series")
    if model not in names:
        raise ValueError(
            f"Unknown foundation model {model!r}; choose from {', '.join(names)}."
        )
    return model


def forecast_foundation(
    series: np.ndarray,
    horizon: int = 10,
    model: str = DEFAULT_FOUNDATION_MODEL,
    context_length: Optional[int] = None,
    accelerator: str = "cpu",
):
    """Zero-shot Darts foundation-model forecast; only ``model``'s weights are fetched.

    A bounded one-model cache reuses weights across repeated agent/UI calls.
    Loading another configuration evicts the previous model.
    """
    from ts_agents.core.forecasting.catalog import get_series_forecaster

    forecaster = get_series_forecaster(_foundation_model_name(model))
    return forecaster(
        np.asarray(series, dtype=float),
        horizon=horizon,
        context_length=context_length,
        accelerator=accelerator,
    )


def forecast_panel_from_csv(
    input_path: str,
    freq: str,
    horizon: int = 24,
    methods: Any = "seasonal_naive",
    season_length: int = 24,
    n_windows: int = 3,
    input_size: Optional[int] = None,
    context_length: Optional[int] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    value_col: str = "y",
    output_dir: Optional[str] = None,
) -> ToolPayload:
    """Run the forecast-panel workflow on a long-format panel file.

    ``methods`` is a comma-separated string or a list. Without ``output_dir`` the
    workflow writes into a fresh directory under the tool artifact directory. The
    workflow payload (metrics, keyed predictions, forecasts, report) is returned
    unchanged.
    """
    from ts_agents.workflows.panel import run_forecast_panel_workflow

    panel_input, options = _prepare_panel_tool_input(
        input_path=input_path, freq=freq, horizon=horizon, methods=methods,
        season_length=season_length, n_windows=n_windows, input_size=input_size,
        context_length=context_length, id_col=id_col, time_col=time_col,
        value_col=value_col, output_dir=output_dir,
    )
    return run_forecast_panel_workflow(panel_input, **options)


def _prepare_panel_tool_input(
    input_path: str, freq: str, horizon: int = 24, methods: Any = "seasonal_naive",
    season_length: int = 24, n_windows: int = 3, input_size: Optional[int] = None,
    context_length: Optional[int] = None, id_col: str = "unique_id",
    time_col: str = "ds", value_col: str = "y", output_dir: Optional[str] = None,
):
    """Read host files into portable records before dispatching a sandbox workflow."""
    from ts_agents.cli.input_parsing import load_panel_input

    if isinstance(methods, str):
        method_list = [method.strip() for method in methods.split(",")]
    else:
        method_list = [str(method).strip() for method in methods or []]
    method_list = [method for method in method_list if method]
    if not method_list:
        raise ValueError("Choose at least one panel forecasting method.")
    panel_input = load_panel_input(
        input_path=input_path,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        freq=freq,
    )
    if output_dir is None:
        output_dir = str(_get_artifact_dir() / f"forecast_panel_{uuid.uuid4().hex[:8]}")
    options: Dict[str, Any] = dict(
        output_dir=output_dir,
        freq=freq,
        horizon=horizon,
        methods=method_list,
        season_length=season_length,
        n_windows=n_windows,
        input_size=input_size,
    )
    if context_length is not None:
        options["context_length"] = context_length
    return panel_input, options
