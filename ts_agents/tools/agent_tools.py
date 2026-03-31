import base64
import io
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional
import uuid

import numpy as np

from ts_agents.contracts import ArtifactRef, ToolPayload
from ts_agents.data_access import get_series as _get_series


_PLOT_LIB = None
_FALLBACK_ARTIFACT_DIR: Optional[Path] = None
_TOOL_ARTIFACT_DIR_ENV = "TS_AGENTS_TOOL_ARTIFACT_DIR"


def _get_plt():
    """Lazy-load matplotlib.pyplot to keep imports light."""
    global _PLOT_LIB
    if _PLOT_LIB is None:
        import matplotlib.pyplot as plt

        _PLOT_LIB = plt
    return _PLOT_LIB


def _get_series_data(
    variable_name: str,
    unique_id: str,
    use_test_data: Optional[bool] = None,
):
    """Load series data helper."""
    return _get_series(
        run_id=unique_id,
        variable_name=variable_name,
        use_test_data=use_test_data,
    )


def _result_data(result):
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return result.to_dict()
    return result


def _series_ref(variable_name: str, unique_id: str) -> dict:
    return {
        "source_type": "bundled_run",
        "run_id": unique_id,
        "variable": variable_name,
    }


def _series_provenance(variable_name: str, unique_id: str) -> dict:
    return {
        "series_ref": _series_ref(variable_name, unique_id),
    }


def _tool_payload(
    *,
    kind: str,
    summary: str,
    data: Any,
    variable_name: Optional[str] = None,
    unique_id: Optional[str] = None,
    artifacts: Optional[list] = None,
    warnings: Optional[list] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> ToolPayload:
    if provenance is None and variable_name is not None and unique_id is not None:
        provenance = _series_provenance(variable_name, unique_id)
    return ToolPayload(
        kind=kind,
        summary=summary,
        data=data,
        artifacts=artifacts or [],
        warnings=warnings or [],
        provenance=provenance or {},
    )


# Legacy helper to format plots for agents (base64 compatibility path).
def _create_plot_response(buf: io.BytesIO) -> str:
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"\n[IMAGE_DATA:{img_base64}]"


def _sanitize_artifact_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "plot"


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


def _create_plot_artifact(
    buf: io.BytesIO,
    *,
    stem: str,
    description: str,
    created_by: str,
) -> ArtifactRef:
    artifact_dir = _get_artifact_dir()
    artifact_path = artifact_dir / (
        f"{_sanitize_artifact_stem(stem)}_{uuid.uuid4().hex[:8]}.png"
    )
    artifact_path.write_bytes(buf.read())
    return ArtifactRef(
        kind="image",
        path=str(artifact_path),
        mime_type="image/png",
        description=description,
        created_by=created_by,
    )


def _finalize_plot_artifact(
    plt,
    fig,
    *,
    stem: str,
    description: str,
    created_by: str,
) -> ArtifactRef:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return _create_plot_artifact(
        buf,
        stem=stem,
        description=description,
        created_by=created_by,
    )


def _multi_series_provenance(series_refs: list[dict]) -> dict:
    return {"series_refs": series_refs}

def stl_decompose_with_data(
    variable_name: str,
    unique_id: str,
    period: Optional[int] = None,
    robust: bool = True,
) -> ToolPayload:
    """Decompose time series using STL with automatic data loading."""
    from ts_agents.core.decomposition import stl_decompose

    series = _get_series_data(variable_name, unique_id)
    result = stl_decompose(series, period=period, robust=robust)
    plt = _get_plt()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(series, label="Original")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"STL Decomposition: {variable_name}")
    axes[1].plot(result.trend, label="Trend", color="orange")
    axes[1].legend(loc="upper left")
    axes[2].plot(
        result.seasonal,
        label=f"Seasonal (period={result.period})",
        color="green",
    )
    axes[2].legend(loc="upper left")
    axes[3].plot(result.residual, label="Residual", color="red")
    axes[3].legend(loc="upper left")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_stl",
        description=f"STL decomposition plot for {variable_name} ({unique_id}).",
        created_by="stl_decompose_with_data",
    )
    data = _result_data(result)
    if isinstance(data, dict):
        data["residual_variance"] = float(np.var(result.residual))
    return _tool_payload(
        kind="decomposition",
        summary=(
            f"STL decomposition completed for {variable_name} "
            f"(run {unique_id}) with period {result.period}."
        ),
        data=data,
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def mstl_decompose_with_data(
    variable_name: str,
    unique_id: str,
    periods: list = None,
) -> ToolPayload:
    from ts_agents.core.decomposition import mstl_decompose

    series = _get_series_data(variable_name, unique_id)
    result = mstl_decompose(series, periods=periods)
    plt = _get_plt()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(series, label="Original")
    axes[0].legend()
    axes[1].plot(result.trend, label="Trend", color="orange")
    axes[1].legend()
    axes[2].plot(result.seasonal, label="Seasonal", color="green")
    axes[2].legend()
    axes[3].plot(result.residual, label="Residual", color="red")
    axes[3].legend()
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_mstl",
        description=f"MSTL decomposition plot for {variable_name} ({unique_id}).",
        created_by="mstl_decompose_with_data",
    )
    return _tool_payload(
        kind="decomposition",
        summary=(
            f"MSTL decomposition completed for {variable_name} "
            f"(run {unique_id})."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def holt_winters_decompose_with_data(
    variable_name: str,
    unique_id: str,
    period: Optional[int] = None,
    trend: str = "add",
    seasonal: str = "add",
) -> ToolPayload:
    from ts_agents.core.decomposition import holt_winters_decompose

    series = _get_series_data(variable_name, unique_id)
    result = holt_winters_decompose(
        series,
        period=period,
        trend=trend,
        seasonal=seasonal,
    )
    plt = _get_plt()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(series, label="Original")
    axes[0].legend()
    axes[1].plot(result.trend, label="Trend", color="orange")
    axes[1].legend()
    axes[2].plot(result.seasonal, label="Seasonal", color="green")
    axes[2].legend()
    axes[3].plot(result.residual, label="Residual", color="red")
    axes[3].legend()
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_holt_winters",
        description=(
            f"Holt-Winters decomposition plot for {variable_name} ({unique_id})."
        ),
        created_by="holt_winters_decompose_with_data",
    )
    return _tool_payload(
        kind="decomposition",
        summary=(
            f"Holt-Winters decomposition completed for {variable_name} "
            f"(run {unique_id})."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )

# Forecasting Wrappers

def forecast_arima_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    confidence_level: float = 0.95,
    season_length: Optional[int] = None,
) -> ToolPayload:
    from ts_agents.core.forecasting import forecast_arima
    series = _get_series_data(variable_name, unique_id)
    level_int = int(confidence_level * 100)
    forecast_kwargs = {
        "horizon": horizon,
        "level": [level_int],
    }
    if season_length is not None:
        forecast_kwargs["season_length"] = season_length
    result = forecast_arima(series, **forecast_kwargs)
    data = _result_data(result)
    return _tool_payload(
        kind="forecast",
        summary=(
            f"ARIMA forecast completed for {variable_name} "
            f"(run {unique_id}) with horizon {horizon}."
        ),
        data=data,
        variable_name=variable_name,
        unique_id=unique_id,
    )

def forecast_ets_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> ToolPayload:
    from ts_agents.core.forecasting import forecast_ets
    series = _get_series_data(variable_name, unique_id)
    forecast_kwargs = {"horizon": horizon}
    if season_length is not None:
        forecast_kwargs["season_length"] = season_length
    result = forecast_ets(series, **forecast_kwargs)
    return _tool_payload(
        kind="forecast",
        summary=(
            f"ETS forecast completed for {variable_name} "
            f"(run {unique_id}) with horizon {horizon}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
    )

def forecast_theta_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> ToolPayload:
    from ts_agents.core.forecasting import forecast_theta
    series = _get_series_data(variable_name, unique_id)
    forecast_kwargs = {"horizon": horizon}
    if season_length is not None:
        forecast_kwargs["season_length"] = season_length
    result = forecast_theta(series, **forecast_kwargs)
    return _tool_payload(
        kind="forecast",
        summary=(
            f"Theta forecast completed for {variable_name} "
            f"(run {unique_id}) with horizon {horizon}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
    )


def forecast_seasonal_naive_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> ToolPayload:
    from ts_agents.core.forecasting import forecast_seasonal_naive
    series = _get_series_data(variable_name, unique_id)
    forecast_kwargs = {"horizon": horizon}
    if season_length is not None:
        forecast_kwargs["season_length"] = season_length
    result = forecast_seasonal_naive(series, **forecast_kwargs)
    return _tool_payload(
        kind="forecast",
        summary=(
            f"Seasonal naive forecast completed for {variable_name} "
            f"(run {unique_id}) with horizon {horizon}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
    )

# Pattern Wrappers

def detect_peaks_with_data(
    variable_name: str,
    unique_id: str,
    distance: int = None,
    prominence: float = None,
) -> ToolPayload:
    from ts_agents.core.patterns import detect_peaks

    series = _get_series_data(variable_name, unique_id)
    result = detect_peaks(series, distance=distance, prominence=prominence)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series, label="Series")
    ax.plot(result.peak_indices, series[result.peak_indices], "x", label="Peaks")
    ax.legend()
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_peaks",
        description=f"Peak detection plot for {variable_name} ({unique_id}).",
        created_by="detect_peaks_with_data",
    )
    summary = f"Detected {result.count} peaks in {variable_name} (run {unique_id})."
    if result.count > 0:
        summary += (
            f" Mean spacing: {result.mean_spacing:.2f}; "
            f"regularity: {result.regularity}."
        )
    return _tool_payload(
        kind="patterns",
        summary=summary,
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def analyze_recurrence_with_data(
    variable_name: str,
    unique_id: str,
    threshold: float = None,
) -> ToolPayload:
    from ts_agents.core.patterns import analyze_recurrence

    series = _get_series_data(variable_name, unique_id)
    result = analyze_recurrence(series, threshold=threshold)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(result.recurrence_matrix, cmap="binary", origin="lower")
    ax.set_title(f"Recurrence Plot (thresh={result.threshold:.2f})")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_recurrence",
        description=f"Recurrence plot for {variable_name} ({unique_id}).",
        created_by="analyze_recurrence_with_data",
    )
    return _tool_payload(
        kind="patterns",
        summary=(
            f"Recurrence analysis completed for {variable_name} (run {unique_id}). "
            f"RR={result.recurrence_rate:.4f}, DET={result.determinism:.4f}, "
            f"LAM={result.laminarity:.4f}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def analyze_matrix_profile_with_data(
    variable_name: str,
    unique_id: str,
    window_size: int = 50,
) -> ToolPayload:
    from ts_agents.core.patterns import analyze_matrix_profile

    series = _get_series_data(variable_name, unique_id)
    result = analyze_matrix_profile(series, m=window_size)
    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(series)
    ax1.set_title("Time Series")
    ax2.plot(result.mp_values)
    ax2.set_title("Matrix Profile (Distance to nearest neighbor)")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_matrix_profile",
        description=f"Matrix profile plot for {variable_name} ({unique_id}).",
        created_by="analyze_matrix_profile_with_data",
    )
    summary = (
        f"Matrix profile analysis completed for {variable_name} "
        f"(run {unique_id}) with window size {window_size}."
    )
    if result.motifs:
        motif = result.motifs[0]
        summary += (
            f" Top motif: index {motif.index} matches {motif.neighbor_index} "
            f"(distance {motif.distance:.4f})."
        )
    if result.discords:
        discord = result.discords[0]
        summary += (
            f" Top discord: index {discord.index} "
            f"(distance {discord.distance:.4f})."
        )
    return _tool_payload(
        kind="patterns",
        summary=summary,
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )

# Classification Wrappers (Example: KNN)

def knn_classify_with_data(
    variable_name: str,
    unique_id: str,
    k: int = 1,
) -> ToolPayload:
    # Note: Classification usually requires a training set. 
    # For now, we assume a pre-trained model or a dataset available in context.
    # This is a placeholder since the core classification tools might need more context.
    return _tool_payload(
        kind="classification",
        summary=(
            f"Classification wrapper for {variable_name} (run {unique_id}) is not "
            f"implemented yet."
        ),
        data={"k": k},
        variable_name=variable_name,
        unique_id=unique_id,
        warnings=["Classification requires a training dataset context. Feature pending."],
    )

# Spectral/Analysis Wrappers

def compute_psd_with_data(
    variable_name: str,
    unique_id: str,
    sampling_rate: float = 1.0,
) -> ToolPayload:
    from ts_agents.core.spectral import compute_psd

    series = _get_series_data(variable_name, unique_id)
    result = compute_psd(series, sample_rate=sampling_rate)
    freqs = result.frequencies
    power = result.psd
    dominant_frequency = float(freqs[np.argmax(power)])
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs, power)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Density")
    ax.set_title(f"Power Spectral Density: {variable_name}")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_psd",
        description=f"Power spectral density plot for {variable_name} ({unique_id}).",
        created_by="compute_psd_with_data",
    )
    return _tool_payload(
        kind="spectral",
        summary=(
            f"Power spectral density computed for {variable_name} "
            f"(run {unique_id}). Dominant frequency: {dominant_frequency:.4f}; "
            f"spectral slope: {result.spectral_slope:.4f}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def compute_spectrum_with_data(
    variable_name: str,
    unique_id: str,
    sampling_rate: float = 1.0,
) -> ToolPayload:
    """Backward-compatible alias for the renamed PSD wrapper."""
    return compute_psd_with_data(
        variable_name=variable_name,
        unique_id=unique_id,
        sampling_rate=sampling_rate,
    )


# ---------------------
# New Wrappers
# ---------------------

# Forecasting

def forecast_ensemble_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    models: list = None,
    season_length: Optional[int] = None,
) -> ToolPayload:
    from ts_agents.core.forecasting import forecast_ensemble
    series = _get_series_data(variable_name, unique_id)
    forecast_kwargs = {
        "horizon": horizon,
        "models": models,
    }
    if season_length is not None:
        forecast_kwargs["season_length"] = season_length
    result = forecast_ensemble(series, **forecast_kwargs)
    data = _result_data(result)
    if not isinstance(data, dict):
        data = {}
    data["ensemble_forecast"] = result.get_ensemble()
    if models is not None:
        model_count = len(models)
    else:
        forecasts = data.get("forecasts")
        model_count = len(forecasts) if isinstance(forecasts, dict) and forecasts else None
    count_text = (
        f" with {model_count} model forecasts."
        if model_count is not None
        else "."
    )
    return _tool_payload(
        kind="forecast_comparison",
        summary=(
            f"Ensemble forecast completed for {variable_name} "
            f"(run {unique_id}){count_text}"
        ),
        data=data,
        variable_name=variable_name,
        unique_id=unique_id,
    )

def compare_forecasts_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    models: list = None,
    methods: list = None,
    season_length: Optional[int] = None,
    ) -> ToolPayload:
    from ts_agents.core.forecasting import compare_forecasts
    series = _get_series_data(variable_name, unique_id)
    selected_models = models if models is not None else methods
    compare_kwargs = {
        "horizon": horizon,
        "models": selected_models,
    }
    if season_length is not None:
        compare_kwargs["season_length"] = season_length
    result = compare_forecasts(series, **compare_kwargs)
    best_model = result.get("best_model")
    summary = (
        f"Forecast comparison completed for {variable_name} (run {unique_id}); "
        f"best model by MAE: {best_model or 'n/a'}."
    )
    return _tool_payload(
        kind="forecast_comparison",
        summary=summary,
        data=result,
        variable_name=variable_name,
        unique_id=unique_id,
    )

# Patterns

def count_peaks_with_data(
    variable_name: str,
    unique_id: str,
    distance: int = None,
    prominence: float = None,
) -> ToolPayload:
    from ts_agents.core.patterns import count_peaks

    series = _get_series_data(variable_name, unique_id)
    peak_count = count_peaks(series, distance=distance, prominence=prominence)
    return _tool_payload(
        kind="patterns",
        summary=f"Counted {peak_count} peaks in {variable_name} (run {unique_id}).",
        data={"count": peak_count},
        variable_name=variable_name,
        unique_id=unique_id,
    )


def find_motifs_with_data(
    variable_name: str,
    unique_id: str,
    window_size: int = 50,
    n_motifs: int = 3,
) -> ToolPayload:
    from ts_agents.core.patterns import find_motifs

    series = _get_series_data(variable_name, unique_id)
    motifs = find_motifs(series, m=window_size, max_motifs=n_motifs)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series, color="gray", alpha=0.5, label="Series")
    if motifs:
        motif = motifs[0]
        idx, neighbor = motif.index, motif.neighbor_index
        ax.plot(
            np.arange(idx, idx + window_size),
            series[idx : idx + window_size],
            "r",
            label="Motif 1",
            linewidth=2,
        )
        ax.plot(
            np.arange(neighbor, neighbor + window_size),
            series[neighbor : neighbor + window_size],
            "g",
            label="Neighbor 1",
            linewidth=2,
        )

    ax.legend()
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_motifs",
        description=f"Motif discovery plot for {variable_name} ({unique_id}).",
        created_by="find_motifs_with_data",
    )
    summary = (
        f"Found {len(motifs)} motifs in {variable_name} "
        f"(run {unique_id}) with window size {window_size}."
    )
    if motifs:
        top_motif = motifs[0]
        summary += (
            f" Top motif: index {top_motif.index} matches {top_motif.neighbor_index} "
            f"(distance {top_motif.distance:.4f})."
        )
    return _tool_payload(
        kind="patterns",
        summary=summary,
        data={"motifs": motifs, "window_size": window_size},
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def find_discords_with_data(
    variable_name: str,
    unique_id: str,
    window_size: int = 50,
    n_discords: int = 3,
) -> ToolPayload:
    from ts_agents.core.patterns import find_discords

    series = _get_series_data(variable_name, unique_id)
    discords = find_discords(series, m=window_size, max_discords=n_discords)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series, color="gray", alpha=0.5, label="Series")
    if discords:
        discord = discords[0]
        idx = discord.index
        ax.plot(
            np.arange(idx, idx + window_size),
            series[idx : idx + window_size],
            "r",
            label="Top Discord",
            linewidth=2,
        )

    ax.legend()
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_discords",
        description=f"Discord detection plot for {variable_name} ({unique_id}).",
        created_by="find_discords_with_data",
    )
    summary = (
        f"Found {len(discords)} discords in {variable_name} "
        f"(run {unique_id}) with window size {window_size}."
    )
    if discords:
        top_discord = discords[0]
        summary += (
            f" Top discord: index {top_discord.index} "
            f"(distance {top_discord.distance:.4f})."
        )
    return _tool_payload(
        kind="patterns",
        summary=summary,
        data={"discords": discords, "window_size": window_size},
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )

def segment_changepoint_with_data(
    variable_name: str,
    unique_id: str,
    n_segments: int = None,
    n_changepoints: int = None,
    algorithm: str = "pelt",
    cost_model: str = "rbf",
    penalty: float = None,
    min_size: int = 5,
) -> ToolPayload:
    from ts_agents.core.patterns import segment_changepoint

    series = _get_series_data(variable_name, unique_id)

    resolved_n_segments = n_segments
    if resolved_n_segments is None and n_changepoints is not None:
        resolved_n_segments = int(n_changepoints) + 1

    result = segment_changepoint(
        series,
        n_segments=resolved_n_segments,
        algorithm=algorithm,
        cost_model=cost_model,
        penalty=penalty,
        min_size=min_size,
    )
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series, label="Series")
    for cp in result.changepoints:
        ax.axvline(x=cp, color="r", linestyle="--", alpha=0.7)
    ax.set_title("Changepoint Detection")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_changepoint",
        description=f"Changepoint detection plot for {variable_name} ({unique_id}).",
        created_by="segment_changepoint_with_data",
    )
    return _tool_payload(
        kind="patterns",
        summary=(
            f"Changepoint detection completed for {variable_name} "
            f"(run {unique_id}). Changepoints: {result.changepoints}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def segment_fluss_with_data(
    variable_name: str,
    unique_id: str,
    window_size: int = 50,
    n_segments: int = 3,
) -> ToolPayload:
    from ts_agents.core.patterns import segment_fluss

    series = _get_series_data(variable_name, unique_id)
    result = segment_fluss(series, m=window_size, n_segments=n_segments)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series, label="Series")
    for cp in result.changepoints:
        ax.axvline(x=cp, color="r", linestyle="--", alpha=0.7)
    if result.changepoints:
        ax.legend()
    ax.set_title("FLUSS Segmentation")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_fluss",
        description=f"FLUSS segmentation plot for {variable_name} ({unique_id}).",
        created_by="segment_fluss_with_data",
    )
    return _tool_payload(
        kind="patterns",
        summary=(
            f"FLUSS segmentation completed for {variable_name} "
            f"(run {unique_id}). Changepoints: {result.changepoints}; "
            f"segments: {result.n_segments}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )

# Spectral (Additional)

def detect_periodicity_with_data(
    variable_name: str,
    unique_id: str,
    n_top: int = 5,
) -> ToolPayload:
    from ts_agents.core.spectral import detect_periodicity

    series = _get_series_data(variable_name, unique_id)
    result = detect_periodicity(series, top_n=n_top)
    return _tool_payload(
        kind="spectral",
        summary=(
            f"Periodicity analysis completed for {variable_name} "
            f"(run {unique_id}). Dominant period: {result.dominant_period:.2f}; "
            f"confidence: {result.confidence:.2f}."
        ),
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
    )

def compute_coherence_with_data(
    variable1: str,
    unique_id1: str,
    variable2: str,
    unique_id2: str,
    sample_rate: float = None,
    sampling_rate: float = None,
    fs: float = None,
) -> ToolPayload:
    from ts_agents.core.spectral import compute_coherence

    series1 = _get_series_data(variable1, unique_id1)
    series2 = _get_series_data(variable2, unique_id2)

    resolved_sample_rate = sample_rate
    if resolved_sample_rate is None:
        if sampling_rate is not None:
            resolved_sample_rate = sampling_rate
        elif fs is not None:
            resolved_sample_rate = fs
        else:
            resolved_sample_rate = 1.0

    result = compute_coherence(series1, series2, sample_rate=resolved_sample_rate)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.frequencies, result.coherence)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Coherence")
    ax.set_title(f"Coherence: {variable1} vs {variable2}")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable1}_{unique_id1}_{variable2}_{unique_id2}_coherence",
        description=(
            f"Coherence plot for {variable1} ({unique_id1}) "
            f"vs {variable2} ({unique_id2})."
        ),
        created_by="compute_coherence_with_data",
    )
    return _tool_payload(
        kind="spectral",
        summary=(
            f"Coherence computed between {variable1} ({unique_id1}) and "
            f"{variable2} ({unique_id2}). Mean coherence: "
            f"{float(np.mean(result.coherence)):.4f}."
        ),
        data=_result_data(result),
        artifacts=[artifact],
        provenance=_multi_series_provenance(
            [_series_ref(variable1, unique_id1), _series_ref(variable2, unique_id2)]
        ),
    )

# Statistics

def describe_series_with_data(variable_name: str, unique_id: str) -> ToolPayload:
    from ts_agents.core.statistics import describe_series
    series = _get_series_data(variable_name, unique_id)
    result = describe_series(series)
    return _tool_payload(
        kind="statistics",
        summary=f"Descriptive statistics computed for {variable_name} (run {unique_id}).",
        data=_result_data(result),
        variable_name=variable_name,
        unique_id=unique_id,
    )

def compute_autocorrelation_with_data(
    variable_name: str,
    unique_id: str,
    max_lag: int = None,
) -> ToolPayload:
    from ts_agents.core.statistics import compute_autocorrelation

    series = _get_series_data(variable_name, unique_id)
    acf = compute_autocorrelation(series, max_lag=max_lag)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(acf)), acf)
    ax.set_title("Autocorrelation Function")
    ax.set_xlabel("Lag")
    plt.tight_layout()
    artifact = _finalize_plot_artifact(
        plt,
        fig,
        stem=f"{variable_name}_{unique_id}_acf",
        description=f"Autocorrelation plot for {variable_name} ({unique_id}).",
        created_by="compute_autocorrelation_with_data",
    )
    lag_one = float(acf[1]) if len(acf) > 1 else None
    return _tool_payload(
        kind="statistics",
        summary=(
            f"Autocorrelation computed for {variable_name} "
            f"(run {unique_id})."
            + (f" Lag 1: {lag_one:.4f}." if lag_one is not None else "")
        ),
        data={"autocorrelation": acf},
        variable_name=variable_name,
        unique_id=unique_id,
        artifacts=[artifact],
    )


def compare_series_stats_with_data(variables: list, run_ids: list) -> ToolPayload:
    from ts_agents.core.statistics import compare_series_stats

    if len(variables) != len(run_ids):
        raise ValueError("variables and run_ids lists must have same length")

    series_dict = {}
    series_refs = []
    for var, run_id in zip(variables, run_ids):
        key = f"{var}_{run_id}"
        series_dict[key] = _get_series_data(var, run_id)
        series_refs.append(_series_ref(var, run_id))

    result = compare_series_stats(series_dict)
    return _tool_payload(
        kind="statistics",
        summary=f"Compared descriptive statistics across {len(series_dict)} series.",
        data=result,
        provenance=_multi_series_provenance(series_refs),
    )
