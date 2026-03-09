import importlib
import sys


def test_core_submodules_are_lazy_loaded():
    sys.modules.pop("ts_agents.core", None)
    sys.modules.pop("ts_agents.core.decomposition", None)

    core = importlib.import_module("ts_agents.core")

    assert "ts_agents.core.decomposition" not in sys.modules
    _ = core.decomposition
    assert "ts_agents.core.decomposition" in sys.modules


def test_core_comparison_exports_are_lazy_loaded():
    sys.modules.pop("ts_agents.core", None)
    sys.modules.pop("ts_agents.core.comparison", None)

    core = importlib.import_module("ts_agents.core")

    assert "ts_agents.core.comparison" not in sys.modules
    _ = core.compare_methods
    assert "ts_agents.core.comparison" in sys.modules
