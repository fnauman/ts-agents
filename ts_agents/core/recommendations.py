from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ts_agents.contracts import ForecastRecommendation

_DEFAULT_AVAILABLE_METHODS = ("seasonal_naive", "theta", "ets", "arima")


def recommend_forecasting_method(
    *,
    stats: Any,
    periodicity: Any,
    acf: Sequence[float],
    available_methods: Iterable[str] | None = None,
) -> ForecastRecommendation:
    stats_dict = _as_mapping(stats)
    periodicity_dict = _as_mapping(periodicity)
    methods = _normalize_methods(available_methods)

    length = int(stats_dict.get("length") or 0)
    std = float(stats_dict.get("std") or 0.0)
    dominant_period = float(periodicity_dict.get("dominant_period") or 0.0)
    periodicity_confidence = float(periodicity_dict.get("confidence") or 0.0)
    lag_one = _lag_one_autocorrelation(acf)
    has_detected_seasonality = dominant_period > 1.0 and periodicity_confidence >= 0.2
    seasonal_history_ok = has_detected_seasonality and length >= max(int(round(dominant_period * 2)), 24)
    arima_history_ok = length >= 32
    non_constant_series = std > 0.0

    rationale: list[str] = []
    preconditions_met: list[str] = []
    preconditions_missing: list[str] = []
    alternatives: list[str] = []

    if non_constant_series:
        preconditions_met.append("series_has_variation")
    else:
        preconditions_missing.append("series_has_variation")

    if has_detected_seasonality:
        preconditions_met.append("seasonality_detected")
        rationale.append(
            f"Detected a dominant period near {dominant_period:.1f} with confidence {periodicity_confidence:.2f}."
        )
    else:
        preconditions_missing.append("seasonality_detected")

    if seasonal_history_ok:
        preconditions_met.append("seasonal_history_sufficient")
    elif has_detected_seasonality:
        preconditions_missing.append("seasonal_history_sufficient")
        rationale.append("Seasonality is present but the history is short for more stable seasonal models.")

    if arima_history_ok:
        preconditions_met.append("history_sufficient_for_arima")
    else:
        preconditions_missing.append("history_sufficient_for_arima")

    if abs(lag_one) >= 0.5:
        preconditions_met.append("strong_short_lag_dependence")
        rationale.append(f"Lag-1 autocorrelation is {lag_one:.2f}, indicating meaningful short-range dependence.")
    else:
        preconditions_missing.append("strong_short_lag_dependence")

    choice = "seasonal_naive"
    if has_detected_seasonality and seasonal_history_ok:
        choice = _first_available(("seasonal_naive", "ets", "theta", "arima"), methods)
        alternatives = _present(("ets", "theta", "arima"), methods, exclude={choice})
        rationale.append("Start with a seasonal baseline before expanding to heavier seasonal models.")
    elif abs(lag_one) >= 0.5:
        choice = _first_available(("theta", "arima", "seasonal_naive", "ets"), methods)
        alternatives = _present(("arima", "seasonal_naive", "ets"), methods, exclude={choice})
        rationale.append("A low-cost autoregressive-style baseline is a good next step after diagnostics.")
    else:
        choice = _first_available(("theta", "seasonal_naive", "ets", "arima"), methods)
        alternatives = _present(("seasonal_naive", "ets", "arima"), methods, exclude={choice})
        rationale.append("Diagnostics are lightweight, so prefer a robust baseline before complex model search.")

    if choice == "arima" and not arima_history_ok:
        rationale.append("ARIMA remains available but history length is marginal, so treat the recommendation as low-confidence.")
    if choice == "seasonal_naive" and has_detected_seasonality and not seasonal_history_ok:
        rationale.append("Seasonal naive is still the safest baseline when seasonality is detected but evidence is limited.")

    confidence = _recommendation_confidence(
        periodicity_confidence=periodicity_confidence,
        has_detected_seasonality=has_detected_seasonality,
        seasonal_history_ok=seasonal_history_ok,
        lag_one=lag_one,
        non_constant_series=non_constant_series,
    )

    return ForecastRecommendation(
        choice=choice,
        rationale=rationale,
        preconditions_met=preconditions_met,
        preconditions_missing=preconditions_missing,
        confidence=confidence,
        alternatives=alternatives,
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, dict):
            return mapped
    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return {}



def _normalize_methods(methods: Iterable[str] | None) -> tuple[str, ...]:
    if methods is None:
        return _DEFAULT_AVAILABLE_METHODS
    normalized = tuple(str(method) for method in methods if str(method))
    return normalized or _DEFAULT_AVAILABLE_METHODS



def _lag_one_autocorrelation(acf: Sequence[float]) -> float:
    if len(acf) <= 1:
        return 0.0
    value = float(acf[1])
    if not np.isfinite(value):
        return 0.0
    return value



def _first_available(preferred: Sequence[str], available: Sequence[str]) -> str:
    for name in preferred:
        if name in available:
            return name
    return available[0]



def _present(preferred: Sequence[str], available: Sequence[str], *, exclude: set[str]) -> list[str]:
    result: list[str] = []
    for name in preferred:
        if name in exclude or name not in available or name in result:
            continue
        result.append(name)
    for name in available:
        if name in exclude or name in result:
            continue
        result.append(name)
    return result



def _recommendation_confidence(
    *,
    periodicity_confidence: float,
    has_detected_seasonality: bool,
    seasonal_history_ok: bool,
    lag_one: float,
    non_constant_series: bool,
) -> float:
    score = 0.35 if non_constant_series else 0.1
    if has_detected_seasonality:
        score += min(max(periodicity_confidence, 0.0), 0.4)
    if seasonal_history_ok:
        score += 0.15
    score += min(abs(lag_one), 1.0) * 0.1
    return round(min(score, 0.95), 4)
