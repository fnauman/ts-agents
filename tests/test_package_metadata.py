from pathlib import Path
import tomllib
from importlib.metadata import PackageNotFoundError, version

import ts_agents


def test_package_exposes_distribution_version():
    try:
        dist_version = version("ts-agents")
    except PackageNotFoundError:
        assert ts_agents.__version__ == "0.0.0.dev0"
    else:
        assert ts_agents.__version__ == dist_version


def test_project_metadata_includes_release_hygiene_fields():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    classifiers = set(project["classifiers"])

    assert any(author.get("email") for author in project["authors"])
    assert project["license"] == "MIT"
    assert "Intended Audience :: Developers" in classifiers
    assert "Intended Audience :: Science/Research" in classifiers
    assert "Typing :: Typed" in classifiers
