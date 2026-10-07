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
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    classifiers = set(project["classifiers"])

    assert any(author.get("email") for author in project["authors"])
    assert project["license"] == "MIT"
    assert "Intended Audience :: Developers" in classifiers
    assert "Intended Audience :: Science/Research" in classifiers
    assert "Typing :: Typed" in classifiers


def test_project_metadata_defines_dependency_extras():
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    extras = project["optional-dependencies"]
    dependencies = set(project["dependencies"])

    assert "ui" in extras
    assert "agents" in extras
    assert "forecasting" in extras
    assert "ml" in extras
    assert "neural" in extras
    assert "decomposition" in extras
    assert "patterns" in extras
    assert "classification" in extras
    assert "foundation" in extras
    assert "recommended" in extras
    assert "all" in extras

    assert not any(dep.startswith("gradio") for dep in dependencies)
    assert not any(dep.startswith("langchain>=") for dep in dependencies)
    assert not any(dep.startswith("statsforecast") for dep in dependencies)
    all_requirements = list(dependencies) + [dep for group in extras.values() for dep in group]
    assert not any(dep.startswith("chronos-forecasting") for dep in all_requirements)
    assert any(dep.startswith("darts[torch]") for dep in extras["foundation"])
    assert any(dep.startswith("darts[torch]") for dep in extras["all"])
    assert any(dep.startswith("torch") for dep in extras["foundation"])
    assert not any(dep.startswith("darts") for dep in dependencies)
    assert not any(dep.startswith("darts") for dep in extras["recommended"])
    assert set(extras["foundation"]).issubset(set(extras["all"]))
    assert set(extras["ml"]).issubset(set(extras["all"]))
    assert set(extras["neural"]).issubset(set(extras["all"]))
    assert not any(dep.startswith(("mlforecast", "neuralforecast", "lightgbm")) for dep in dependencies)
    assert not any(dep.startswith(("mlforecast", "neuralforecast", "lightgbm")) for dep in extras["recommended"])
    assert project["scripts"]["ts-agents-ui"] == "ts_agents.ui.entrypoint:main"
