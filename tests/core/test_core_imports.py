import importlib
import sys


def test_core_submodules_are_lazy_loaded():
    sys.modules.pop("src.core", None)
    sys.modules.pop("src.core.decomposition", None)

    core = importlib.import_module("src.core")

    assert "src.core.decomposition" not in sys.modules
    _ = core.decomposition
    assert "src.core.decomposition" in sys.modules


def test_core_comparison_exports_are_lazy_loaded():
    sys.modules.pop("src.core", None)
    sys.modules.pop("src.core.comparison", None)

    core = importlib.import_module("src.core")

    assert "src.core.comparison" not in sys.modules
    _ = core.compare_methods
    assert "src.core.comparison" in sys.modules
