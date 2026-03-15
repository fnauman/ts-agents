import base64
import io
from typing import Optional

import numpy as np

from ts_agents.core.decomposition import stl_decompose
from ts_agents.data_access import get_series as _get_series


_PLOT_LIB = None


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
    
# Helper to format plots for agents (base64)
def _create_plot_response(buf: io.BytesIO) -> str:
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"\n[IMAGE_DATA:{img_base64}]"

def stl_decompose_with_data(
    variable_name: str,
    unique_id: str,
    period: Optional[int] = None,
    robust: bool = True
) -> str:
    """
    Decompose time series using STL (Season-Trend LOESS) with automatic data loading and plotting.
    """
    try:
        # Load data
        series = _get_series_data(variable_name, unique_id)
        
        # Call core function
        result = stl_decompose(series, period=period, robust=robust)
        
        # Format text output
        output = f"STL Decomposition for {variable_name} (run {unique_id}):\n"
        output += f"- Period used: {result.period}\n"
        output += f"- Residual variance: {np.var(result.residual):.4f}\n"
        
        # Create plot
        plt = _get_plt()
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        
        axes[0].plot(series, label='Original')
        axes[0].legend(loc='upper left')
        axes[0].set_title(f'STL Decomposition: {variable_name}')
        
        axes[1].plot(result.trend, label='Trend', color='orange')
        axes[1].legend(loc='upper left')
        
        axes[2].plot(result.seasonal, label=f'Seasonal (period={result.period})', color='green')
        axes[2].legend(loc='upper left')
        
        axes[3].plot(result.residual, label='Residual', color='red')
        axes[3].legend(loc='upper left')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        output += _create_plot_response(buf)
        return output
        
    except Exception as e:
        return f"Error in stl_decompose: {str(e)}"

def mstl_decompose_with_data(variable_name: str, unique_id: str, periods: list = None) -> str:
    from ts_agents.core.decomposition import mstl_decompose
    try:
        series = _get_series_data(variable_name, unique_id)
        result = mstl_decompose(series, periods=periods)
        
        output = f"MSTL Decomposition for {variable_name} (run {unique_id}):\n"
        output += f"- Periods used: {result.period}\n"
        
        # Plotting MSTL (similar to STL)
        plt = _get_plt()
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(series, label='Original')
        axes[0].legend()
        axes[1].plot(result.trend, label='Trend', color='orange')
        axes[1].legend()
        axes[2].plot(result.seasonal, label='Seasonal', color='green')
        axes[2].legend()
        axes[3].plot(result.residual, label='Residual', color='red')
        axes[3].legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in MSTL: {str(e)}"

def holt_winters_decompose_with_data(variable_name: str, unique_id: str, period: Optional[int] = None, trend: str = 'add', seasonal: str = 'add') -> str:
    from ts_agents.core.decomposition import holt_winters_decompose
    try:
        series = _get_series_data(variable_name, unique_id)
        result = holt_winters_decompose(series, period=period, trend=trend, seasonal=seasonal)
        
        output = f"Holt-Winters for {variable_name} (run {unique_id}):\n"
        
        plt = _get_plt()
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(series, label='Original')
        axes[0].legend()
        axes[1].plot(result.trend, label='Trend', color='orange')
        axes[1].legend()
        axes[2].plot(result.seasonal, label='Seasonal', color='green')
        axes[2].legend()
        axes[3].plot(result.residual, label='Residual', color='red')
        axes[3].legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Holt-Winters: {str(e)}"

# Forecasting Wrappers

def forecast_arima_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    confidence_level: float = 0.95,
    season_length: Optional[int] = None,
) -> str:
    from ts_agents.core.forecasting import forecast_arima
    try:
        series = _get_series_data(variable_name, unique_id)
        # Convert confidence_level 0.95 to level [95]
        level_int = int(confidence_level * 100)
        forecast_kwargs = {
            "horizon": horizon,
            "level": [level_int],
        }
        if season_length is not None:
            forecast_kwargs["season_length"] = season_length
        result = forecast_arima(series, **forecast_kwargs)
        
        output = f"ARIMA Forecast for {variable_name} (run {unique_id}):\n"
        output += f"- Horizon: {horizon}\n"
        output += f"- Forecast Values (next 5): {result.forecast[:5]}\n"

        # Plotting
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot last part of history
        hist = series[-min(len(series), 100):]
        ax.plot(np.arange(len(hist)), hist, label='History')
        # Plot forecast
        x_pred = np.arange(len(hist), len(hist) + horizon)
        ax.plot(x_pred, result.forecast, label='Forecast')
        if result.lower_bound is not None:
             ax.fill_between(x_pred, result.lower_bound, result.upper_bound, color='gray', alpha=0.2, label=f'{level_int}% CI')
        
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in ARIMA: {str(e)}"

def forecast_ets_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> str:
    from ts_agents.core.forecasting import forecast_ets
    try:
        series = _get_series_data(variable_name, unique_id)
        forecast_kwargs = {"horizon": horizon}
        if season_length is not None:
            forecast_kwargs["season_length"] = season_length
        result = forecast_ets(series, **forecast_kwargs)
        
        output = f"ETS Forecast for {variable_name} (run {unique_id}):\n"
        output += f"First 5 predictions: {result.forecast[:5]}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = series[-min(len(series), 100):]
        ax.plot(np.arange(len(hist)), hist, label='History')
        x_pred = np.arange(len(hist), len(hist) + horizon)
        ax.plot(x_pred, result.forecast, label='Forecast')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in ETS: {str(e)}"

def forecast_theta_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> str:
    from ts_agents.core.forecasting import forecast_theta
    try:
        series = _get_series_data(variable_name, unique_id)
        forecast_kwargs = {"horizon": horizon}
        if season_length is not None:
            forecast_kwargs["season_length"] = season_length
        result = forecast_theta(series, **forecast_kwargs)

        output = f"Theta Forecast for {variable_name} (run {unique_id}):\n"
        output += f"First 5 predictions: {result.forecast[:5]}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = series[-min(len(series), 100):]
        ax.plot(np.arange(len(hist)), hist, label='History')
        x_pred = np.arange(len(hist), len(hist) + horizon)
        ax.plot(x_pred, result.forecast, label='Forecast')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Theta: {str(e)}"


def forecast_seasonal_naive_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    season_length: Optional[int] = None,
) -> str:
    from ts_agents.core.forecasting import forecast_seasonal_naive
    try:
        series = _get_series_data(variable_name, unique_id)
        forecast_kwargs = {"horizon": horizon}
        if season_length is not None:
            forecast_kwargs["season_length"] = season_length
        result = forecast_seasonal_naive(series, **forecast_kwargs)

        output = f"Seasonal Naive Forecast for {variable_name} (run {unique_id}):\n"
        output += f"- Horizon: {horizon}\n"
        output += f"- Season length: {season_length or 1}\n"
        output += f"- First 5 predictions: {result.forecast[:5]}\n"

        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = series[-min(len(series), 100):]
        ax.plot(np.arange(len(hist)), hist, label='History')
        x_pred = np.arange(len(hist), len(hist) + horizon)
        ax.plot(x_pred, result.forecast, label='Forecast')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Seasonal Naive: {str(e)}"

# Pattern Wrappers

def detect_peaks_with_data(variable_name: str, unique_id: str, distance: int = None, prominence: float = None) -> str:
    from ts_agents.core.patterns import detect_peaks
    try:
        series = _get_series_data(variable_name, unique_id)
        result = detect_peaks(series, distance=distance, prominence=prominence)
        
        output = f"Peaks detected in {variable_name}: {result.count}\n"
        if result.count > 0:
            output += f"Mean spacing: {result.mean_spacing:.2f}\n"
            output += f"Regularity: {result.regularity}\n"
            
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series, label='Series')
        ax.plot(result.peak_indices, series[result.peak_indices], 'x', label='Peaks')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in detect_peaks: {str(e)}"

def analyze_recurrence_with_data(variable_name: str, unique_id: str, threshold: float = None) -> str:
    from ts_agents.core.patterns import analyze_recurrence
    try:
        series = _get_series_data(variable_name, unique_id)
        result = analyze_recurrence(series, threshold=threshold)
        
        output = f"Recurrence Analysis for {variable_name}:\n"
        output += f"RR (Recurrence Rate): {result.recurrence_rate:.4f}\n"
        output += f"DET (Determinism): {result.determinism:.4f}\n"
        output += f"LAM (Laminarity): {result.laminarity:.4f}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(result.recurrence_matrix, cmap='binary', origin='lower')
        ax.set_title(f"Recurrence Plot (thresh={result.threshold:.2f})")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in recurrence: {str(e)}"

def analyze_matrix_profile_with_data(variable_name: str, unique_id: str, window_size: int = 50) -> str:
    from ts_agents.core.patterns import analyze_matrix_profile
    try:
        series = _get_series_data(variable_name, unique_id)
        result = analyze_matrix_profile(series, m=window_size)
        
        output = f"Matrix Profile for {variable_name} (m={window_size}):\n"
        if result.motifs:
            m = result.motifs[0]
            output += f"Top Motif: Index {m.index} matches {m.neighbor_index} (dist {m.distance:.4f})\n"
        if result.discords:
            d = result.discords[0]
            output += f"Top Discord (Anomaly): Index {d.index} (dist {d.distance:.4f})\n"
            
        plt = _get_plt()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(series)
        ax1.set_title("Time Series")
        ax2.plot(result.mp_values)
        ax2.set_title("Matrix Profile (Distance to nearest neighbor)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in MP: {str(e)}"

# Classification Wrappers (Example: KNN)

def knn_classify_with_data(variable_name: str, unique_id: str, k: int = 1) -> str:
    # Note: Classification usually requires a training set. 
    # For now, we assume a pre-trained model or a dataset available in context.
    # This is a placeholder since the core classification tools might need more context.
    return "Classification requires a training dataset context. Feature pending."

# Spectral/Analysis Wrappers

def compute_psd_with_data(variable_name: str, unique_id: str, sampling_rate: float = 1.0) -> str:
    from ts_agents.core.spectral import compute_psd
    try:
        series = _get_series_data(variable_name, unique_id)
        result = compute_psd(series, sample_rate=sampling_rate)

        output = f"Power Spectral Density for {variable_name}:\n"
        freqs = result.frequencies
        power = result.psd
        output += f"Dominant Frequency: {freqs[np.argmax(power)]:.4f}\n"
        output += f"Spectral Slope: {result.spectral_slope:.4f}\n"

        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(freqs, power)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power Density")
        ax.set_title(f"Power Spectral Density: {variable_name}")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in PSD: {str(e)}"


def compute_spectrum_with_data(variable_name: str, unique_id: str, sampling_rate: float = 1.0) -> str:
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
) -> str:
    from ts_agents.core.forecasting import forecast_ensemble
    try:
        series = _get_series_data(variable_name, unique_id)
        forecast_kwargs = {
            "horizon": horizon,
            "models": models,
        }
        if season_length is not None:
            forecast_kwargs["season_length"] = season_length
        result = forecast_ensemble(series, **forecast_kwargs)
        ensemble_forecast = result.get_ensemble()
        
        output = f"Ensemble Forecast for {variable_name}:\n"
        output += f"First 5 predictions: {ensemble_forecast[:5]}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = series[-min(len(series), 100):]
        ax.plot(np.arange(len(hist)), hist, label='History')
        x_pred = np.arange(len(hist), len(hist) + horizon)
        ax.plot(x_pred, ensemble_forecast, label='Ensemble Forecast', linewidth=2)
        
        # Plot individual models if available
        # Assuming result has a way to access individual forecasts, but for now just plotting ensemble
        
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Ensemble: {str(e)}"

def compare_forecasts_with_data(
    variable_name: str,
    unique_id: str,
    horizon: int = 10,
    models: list = None,
    methods: list = None,
    season_length: Optional[int] = None,
    ) -> str:
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

    output = f"Forecast Comparison for {variable_name}:\n"
    output += str(result)
    return output

# Patterns

def count_peaks_with_data(variable_name: str, unique_id: str, distance: int = None, prominence: float = None) -> int:
    from ts_agents.core.patterns import count_peaks
    try:
        series = _get_series_data(variable_name, unique_id)
        return count_peaks(series, distance=distance, prominence=prominence)
    except Exception as e:
        # Return -1 or error string? Wrapper usually returns string for LLM, but registry says int.
        # Wrappers should return string for LLM readability mostly.
        return f"Error counting peaks: {str(e)}"

def find_motifs_with_data(variable_name: str, unique_id: str, window_size: int = 50, n_motifs: int = 3) -> str:
    from ts_agents.core.patterns import find_motifs
    try:
        series = _get_series_data(variable_name, unique_id)
        # Map parameters: window_size -> m, n_motifs -> max_motifs
        motifs = find_motifs(series, m=window_size, max_motifs=n_motifs)
        
        output = f"Top {len(motifs)} Motifs for {variable_name} (m={window_size}):\n"
        for i, m in enumerate(motifs):
            output += f"{i+1}. Index {m.index} matches {m.neighbor_index} (dist {m.distance:.4f})\n"
            
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series, color='gray', alpha=0.5, label='Series')
        # Highlight top motif
        if motifs:
            m = motifs[0]
            idx, neighbor = m.index, m.neighbor_index
            ax.plot(np.arange(idx, idx+window_size), series[idx:idx+window_size], 'r', label='Motif 1', linewidth=2)
            ax.plot(np.arange(neighbor, neighbor+window_size), series[neighbor:neighbor+window_size], 'g', label='Neighbor 1', linewidth=2)
        
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Motifs: {str(e)}"

def find_discords_with_data(variable_name: str, unique_id: str, window_size: int = 50, n_discords: int = 3) -> str:
    from ts_agents.core.patterns import find_discords
    try:
        series = _get_series_data(variable_name, unique_id)
        # Map parameters: window_size -> m, n_discords -> max_discords
        discords = find_discords(series, m=window_size, max_discords=n_discords)
        
        output = f"Top {len(discords)} Discords (Anomalies) for {variable_name} (m={window_size}):\n"
        for i, d in enumerate(discords):
            output += f"{i+1}. Index {d.index} (dist {d.distance:.4f})\n"
            
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series, color='gray', alpha=0.5, label='Series')
        if discords:
            d = discords[0]
            idx = d.index
            ax.plot(np.arange(idx, idx+window_size), series[idx:idx+window_size], 'r', label='Top Discord', linewidth=2)
            
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Discords: {str(e)}"

def segment_changepoint_with_data(
    variable_name: str,
    unique_id: str,
    n_segments: int = None,
    n_changepoints: int = None,
    algorithm: str = "pelt",
    cost_model: str = "rbf",
    penalty: float = None,
    min_size: int = 5,
) -> str:
    from ts_agents.core.patterns import segment_changepoint
    try:
        series = _get_series_data(variable_name, unique_id)

        # Backward-compatibility: `n_changepoints` maps to `n_segments = n_changepoints + 1`.
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
        
        output = f"Changepoints detected at indices: {result.changepoints}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series, label='Series')
        for cp in result.changepoints:
            ax.axvline(x=cp, color='r', linestyle='--', alpha=0.7)
        ax.set_title("Changepoint Detection")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Changepoint: {str(e)}"

def segment_fluss_with_data(variable_name: str, unique_id: str, window_size: int = 50, n_segments: int = 3) -> str:
    from ts_agents.core.patterns import segment_fluss
    try:
        series = _get_series_data(variable_name, unique_id)
        result = segment_fluss(series, m=window_size, n_segments=n_segments)

        output = f"FLUSS segmentation for {variable_name}:\n"
        output += f"Changepoints detected at indices: {result.changepoints}\n"
        output += f"Segments: {result.n_segments}\n"
        if result.segment_stats:
            output += "Segment statistics:\n"
            for idx, stats in enumerate(result.segment_stats, start=1):
                output += (
                    f"- Segment {idx}: start={stats.get('start', 'n/a')}, "
                    f"end={stats.get('end', 'n/a')}, "
                    f"length={stats.get('length', 'n/a')}, "
                    f"mean={stats.get('mean', float('nan')):.4f}, "
                    f"std={stats.get('std', float('nan')):.4f}\n"
                )

        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series, label="Series")
        for cp in result.changepoints:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.7)
        if result.changepoints:
            ax.legend()
        ax.set_title("FLUSS Segmentation")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in FLUSS segmentation: {str(e)}"

# Spectral (Additional)

def detect_periodicity_with_data(variable_name: str, unique_id: str, n_top: int = 5) -> str:
    from ts_agents.core.spectral import detect_periodicity
    try:
        series = _get_series_data(variable_name, unique_id)
        # Map parameters: n_top -> top_n
        result = detect_periodicity(series, top_n=n_top)
        
        output = f"Periodicity Analysis for {variable_name}:\n"
        output += f"Dominant Period: {result.dominant_period:.2f}\n"
        output += f"Confidence: {result.confidence:.2f}\n"
        output += f"Top periods: {result.top_periods}\n"
        return output
    except Exception as e:
        return f"Error in Periodicity: {str(e)}"

def compute_coherence_with_data(
    variable1: str,
    unique_id1: str,
    variable2: str,
    unique_id2: str,
    sample_rate: float = None,
    sampling_rate: float = None,
    fs: float = None,
) -> str:
    from ts_agents.core.spectral import compute_coherence
    try:
        series1 = _get_series_data(variable1, unique_id1)
        series2 = _get_series_data(variable2, unique_id2)

        # Backward-compatible aliases: sample_rate (preferred), sampling_rate, fs.
        resolved_sample_rate = sample_rate
        if resolved_sample_rate is None:
            if sampling_rate is not None:
                resolved_sample_rate = sampling_rate
            elif fs is not None:
                resolved_sample_rate = fs
            else:
                resolved_sample_rate = 1.0

        result = compute_coherence(series1, series2, sample_rate=resolved_sample_rate)
        
        output = f"Coherence between {variable1} (Run {unique_id1}) and {variable2} (Run {unique_id2}):\n"
        output += f"Mean Coherence: {np.mean(result.coherence):.4f}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result.frequencies, result.coherence)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Coherence')
        ax.set_title(f"Coherence: {variable1} vs {variable2}")
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in Coherence: {str(e)}"

# Statistics

def describe_series_with_data(variable_name: str, unique_id: str) -> str:
    from ts_agents.core.statistics import describe_series
    try:
        series = _get_series_data(variable_name, unique_id)
        result = describe_series(series)
        return f"Statistics for {variable_name} (Run {unique_id}):\n{result}"
    except Exception as e:
        return f"Error in Stats: {str(e)}"

def compute_autocorrelation_with_data(variable_name: str, unique_id: str, max_lag: int = None) -> str:
    from ts_agents.core.statistics import compute_autocorrelation
    try:
        series = _get_series_data(variable_name, unique_id)
        acf = compute_autocorrelation(series, max_lag=max_lag)
        
        output = f"Autocorrelation (ACF) for {variable_name}:\n"
        output += f"Lag 1: {acf[1] if len(acf)>1 else 'N/A'}\n"
        
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(np.arange(len(acf)), acf)
        ax.set_title("Autocorrelation Function")
        ax.set_xlabel("Lag")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        output += _create_plot_response(buf)
        return output
    except Exception as e:
        return f"Error in ACF: {str(e)}"

def compare_series_stats_with_data(variables: list, run_ids: list) -> str:
    from ts_agents.core.statistics import compare_series_stats
    try:
        # Check inputs
        if len(variables) != len(run_ids):
             return "Error: variables and run_ids lists must have same length."
        
        series_dict = {}
        for var, run_id in zip(variables, run_ids):
            key = f"{var}_{run_id}"
            series_dict[key] = _get_series_data(var, run_id)
            
        result = compare_series_stats(series_dict)
        return f"Comparison Statistics:\n{result}"
    except Exception as e:
        return f"Error in Compare Stats: {str(e)}"
