from importlib.metadata import version

import ts_agents


def test_package_exposes_distribution_version():
    assert ts_agents.__version__ == version("ts-agents")
