import pytest

from ts_agents.core.recommendations import recommend_forecasting_method


@pytest.mark.parametrize(
    ("stats", "periodicity", "acf", "available_methods", "expected_choice"),
    [
        (
            {"length": 96, "std": 1.2},
            {"dominant_period": 24.0, "confidence": 0.72},
            [1.0, 0.31, 0.12],
            ["seasonal_naive", "theta", "ets", "arima"],
            "seasonal_naive",
        ),
        (
            {"length": 48, "std": 0.8},
            {"dominant_period": 0.0, "confidence": 0.05},
            [1.0, 0.86, 0.55],
            ["theta", "arima"],
            "theta",
        ),
        (
            {"length": 18, "std": 0.4},
            {"dominant_period": 12.0, "confidence": 0.4},
            [1.0, 0.2, 0.1],
            ["seasonal_naive"],
            "seasonal_naive",
        ),
    ],
)
def test_recommend_forecasting_method_selects_expected_baseline(
    stats,
    periodicity,
    acf,
    available_methods,
    expected_choice,
):
    recommendation = recommend_forecasting_method(
        stats=stats,
        periodicity=periodicity,
        acf=acf,
        available_methods=available_methods,
    )

    assert recommendation.choice == expected_choice
    assert recommendation.confidence is not None
    assert 0.0 < recommendation.confidence <= 0.95
    assert recommendation.rationale


def test_recommend_forecasting_method_marks_met_and_missing_preconditions():
    recommendation = recommend_forecasting_method(
        stats={"length": 72, "std": 1.0},
        periodicity={"dominant_period": 0.0, "confidence": 0.03},
        acf=[1.0, 0.77, 0.42],
        available_methods=["theta", "arima"],
    )

    assert recommendation.choice == "theta"
    assert "strong_short_lag_dependence" in recommendation.preconditions_met
    assert "seasonality_detected" in recommendation.preconditions_missing
    assert "arima" in recommendation.alternatives


def test_recommend_forecasting_method_reports_low_information_series():
    recommendation = recommend_forecasting_method(
        stats={"length": 12, "std": 0.0},
        periodicity={"dominant_period": 0.0, "confidence": 0.0},
        acf=[1.0],
        available_methods=["seasonal_naive"],
    )

    assert recommendation.choice == "seasonal_naive"
    assert "series_has_variation" in recommendation.preconditions_missing
    assert recommendation.confidence is not None
    assert recommendation.confidence < 0.4
