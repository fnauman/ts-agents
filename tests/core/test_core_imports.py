import importlib
import sys


def test_core_submodules_are_lazy_loaded():
    sys.modules.pop("ts_agents.core", None)
    sys.modules.pop("ts_agents.core.decomposition", None)

    core = importlib.import_module("ts_agents.core")

    assert "ts_agents.core.decomposition" not in sys.modules
    _ = core.decomposition
    assert "ts_agents.core.decomposition" in sys.modules


def test_core_forecasting_exports_are_lazy_loaded():
    sys.modules.pop("ts_agents.core.forecasting", None)
    sys.modules.pop("ts_agents.core.forecasting.statistical", None)

    forecasting = importlib.import_module("ts_agents.core.forecasting")

    assert "ts_agents.core.forecasting.statistical" not in sys.modules
    _ = forecasting.forecast_arima
    assert "ts_agents.core.forecasting.statistical" in sys.modules


def test_agents_package_import_is_lazy():
    sys.modules.pop("ts_agents.agents", None)
    sys.modules.pop("ts_agents.agents.simple", None)
    sys.modules.pop("ts_agents.agents.deep", None)
    sys.modules.pop("ts_agents.agents.benchmarks", None)

    agents = importlib.import_module("ts_agents.agents")

    assert "ts_agents.agents.simple" not in sys.modules
    assert hasattr(agents, "__all__")


def test_core_comparison_exports_are_lazy_loaded():
    sys.modules.pop("ts_agents.core", None)
    sys.modules.pop("ts_agents.core.comparison", None)

    core = importlib.import_module("ts_agents.core")

    assert "ts_agents.core.comparison" not in sys.modules
    _ = core.compare_methods
    assert "ts_agents.core.comparison" in sys.modules
