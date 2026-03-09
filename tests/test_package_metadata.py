from importlib.metadata import PackageNotFoundError, version

import ts_agents


def test_package_exposes_distribution_version():
    try:
        dist_version = version("ts-agents")
    except PackageNotFoundError:
        assert ts_agents.__version__ == "0.0.0.dev0"
    else:
        assert ts_agents.__version__ == dist_version
