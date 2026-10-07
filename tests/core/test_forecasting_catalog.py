import importlib.util
import json
import re

import numpy as np
import pytest

from ts_agents.core.forecasting import catalog


def test_method_names_are_unique_and_surfaces_known():
    names = [method.name for method in catalog.METHODS]
    assert len(names) == len(set(names))
    for method in catalog.METHODS:
        assert set(method.surfaces) <= set(catalog.SURFACES)
        assert method.surfaces


def test_surface_listings_follow_catalog_order():
    assert catalog.methods_for("series") == [
        "seasonal_naive", "arima", "ets", "theta",
        "chronos2_small", "chronos2", "timesfm2p5", "patchtst_fm",
    ]
    assert catalog.methods_for("panel") == [
        "seasonal_naive", "lightgbm", "histgbm", "nhits",
        "chronos2_small", "chronos2", "timesfm2p5", "patchtst_fm",
    ]
    assert catalog.methods_for("autoresearch") == catalog.foundation_methods()
    with pytest.raises(ValueError, match="surface"):
        catalog.methods_for("ui")


def test_foundation_entries_are_pinned_darts_models():
    assert catalog.DEFAULT_FOUNDATION_MODEL == "chronos2_small"
    assert catalog.DEFAULT_FOUNDATION_MODEL in catalog.foundation_methods("panel")
    for name in catalog.foundation_methods():
        method = catalog.get_method(name)
        spec = method.foundation
        assert method.backend == "darts" and method.zero_shot
        assert method.extra == catalog.FOUNDATION_EXTRA
        assert method.modules == catalog.FOUNDATION_MODULES
        assert re.fullmatch(r"[0-9a-f]{40}", spec.hub_model_revision)
        assert spec.default_context < spec.max_context
        assert spec.license == "apache-2.0"
    assert set(catalog.FOUNDATION_DISTRIBUTIONS) == set(catalog.FOUNDATION_MODULES)
    checkpoints = catalog.foundation_checkpoints("panel")
    assert list(checkpoints) == catalog.foundation_methods("panel")
    assert checkpoints["chronos2_small"]["hub_model_name"] == "autogluon/chronos-2-small"
    json.dumps(checkpoints)


def test_timesfm_long_horizon_overrides_only_above_native_cap():
    spec = catalog.get_method("timesfm2p5").foundation
    assert spec.constructor_overrides(128) == {}
    assert spec.constructor_overrides(200) == {"use_longer_projection_head": True}
    assert catalog.get_method("chronos2").foundation.constructor_overrides(900) == {}


def test_required_extras_are_sorted_unique_and_skip_base_methods():
    assert catalog.required_extras_for(["seasonal_naive", "lightgbm", "chronos2"]) == [
        "foundation",
        "ml",
    ]
    assert catalog.required_extras_for("nhits, timesfm2p5,histgbm") == ["foundation", "ml", "neural"]
    assert catalog.required_extras_for(["seasonal_naive"]) == []
    assert catalog.required_extras_for(None) == []
    assert catalog.method_extras("series")["arima"] == "forecasting"
    assert catalog.method_extras("panel")["seasonal_naive"] is None


def test_install_hint_reports_only_missing_extras(monkeypatch):
    real_find_spec = importlib.util.find_spec
    missing = {"darts", "huggingface_hub"}

    def fake_find_spec(name, *args, **kwargs):
        if name in missing:
            return None
        if name in {"mlforecast", "lightgbm", "torch"}:
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert catalog.missing_modules("chronos2_small") == ["darts", "huggingface_hub"]
    assert not catalog.is_available("chronos2_small")
    assert catalog.is_available("lightgbm")
    assert catalog.install_hint_for(["seasonal_naive", "lightgbm", "chronos2_small"]) == (
        "Install `ts-agents[foundation]`."
    )
    assert catalog.install_hint_for(["seasonal_naive", "lightgbm"]) is None


def test_unknown_method_raises_value_error_listing_names():
    with pytest.raises(ValueError, match="chronos2_small"):
        catalog.get_method("chronos-t5-tiny")
    with pytest.raises(ValueError):
        catalog.get_series_forecaster("not-a-method")
    with pytest.raises(ValueError, match="panel-only"):
        catalog.get_series_forecaster("lightgbm")


def test_series_forecaster_drops_unaccepted_options():
    forecaster = catalog.get_series_forecaster("seasonal_naive")
    result = forecaster(
        np.arange(1.0, 9.0), horizon=4, season_length=4, context_length=512, accelerator="gpu"
    )
    assert result.method == "seasonal_naive"
    np.testing.assert_allclose(result.forecast, [5.0, 6.0, 7.0, 8.0])


def test_series_forecaster_for_arima_ignores_context_length(monkeypatch):
    from ts_agents.core.forecasting import statistical

    captured = {}

    def fake_arima(series, horizon=10, level=None, season_length=None):
        captured.update(horizon=horizon, season_length=season_length)
        return "arima-result"

    monkeypatch.setattr(statistical, "forecast_arima", fake_arima)
    forecaster = catalog.get_series_forecaster("arima")
    assert forecaster(np.ones(20), horizon=3, season_length=7, context_length=64) == "arima-result"
    assert captured == {"horizon": 3, "season_length": 7}


def test_series_forecaster_binds_foundation_model(monkeypatch):
    from ts_agents.core.forecasting import foundation

    captured = {}

    def fake_forecast_foundation(series, **kwargs):
        captured.update(kwargs)
        return "fm-result"

    monkeypatch.setattr(foundation, "forecast_foundation", fake_forecast_foundation)
    forecaster = catalog.get_series_forecaster("timesfm2p5")
    assert forecaster(np.ones(20), horizon=5, season_length=12, context_length=64, seed=3) == "fm-result"
    assert captured == {"model": "timesfm2p5", "horizon": 5, "context_length": 64, "seed": 3}
