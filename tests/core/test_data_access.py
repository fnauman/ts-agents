import importlib
import logging
import os

import numpy as np
import pandas as pd


def _setup_test_data(monkeypatch, tmp_path):
    data = pd.DataFrame(
        {
            "unique_id": ["Re200Rm200", "Re200Rm200"],
            "ds": [0, 1],
            "bx001_real": [1.0, 2.0],
            "by001_real": [3.0, 4.0],
        }
    )
    csv_path = tmp_path / "short_real.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setenv("TS_AGENTS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TS_AGENTS_USE_TEST_DATA", "true")
    monkeypatch.setenv("TS_AGENTS_TEST_DATA_FILE", "short_real.csv")

    import ts_agents.config as config
    import ts_agents.data_access as data_access

    importlib.reload(config)
    importlib.reload(data_access)
    data_access.clear_cache()

    return data_access, data


def test_get_series_resolves_alias(monkeypatch, tmp_path):
    data_access, data = _setup_test_data(monkeypatch, tmp_path)

    series = data_access.get_series("Re200Rm200", "y")
    expected = data["by001_real"].values

    assert np.array_equal(series, expected)


def test_load_dataframe_cache(monkeypatch, tmp_path):
    data_access, _ = _setup_test_data(monkeypatch, tmp_path)

    df1 = data_access.load_dataframe("real")
    df2 = data_access.load_dataframe("real")

    assert df1 is df2


def test_infer_data_type(monkeypatch, tmp_path):
    data_access, _ = _setup_test_data(monkeypatch, tmp_path)

    assert data_access.infer_data_type("bx001_real") == "real"
    assert data_access.infer_data_type("bx001_imag") == "imag"
    assert data_access.infer_data_type("unknown") == "real"


def test_config_default_data_dir_is_runtime_resolved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TS_AGENTS_DATA_DIR", raising=False)

    import ts_agents.config as config
    from ts_agents.runtime_paths import resolve_default_data_dir

    importlib.reload(config)

    assert config.DATA_DIR == resolve_default_data_dir()
    assert config.DATA_DIR != tmp_path / "data"


def test_config_import_does_not_load_user_env(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("OPENAI_MODEL=dotenv-model\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    import ts_agents.config as config

    importlib.reload(config)

    assert os.environ.get("OPENAI_MODEL") is None
    assert config.get_openai_model() == "dotenv-model"
    assert os.environ["OPENAI_MODEL"] == "dotenv-model"


def test_config_import_does_not_warn_on_missing_data_dir(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TS_AGENTS_DATA_DIR", raising=False)

    import ts_agents.config as config

    caplog.set_level(logging.WARNING)
    importlib.reload(config)

    assert not caplog.records


def test_config_persistence_dir_is_resolved_on_access(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    monkeypatch.delenv("TS_AGENTS_PERSISTENCE_DIR", raising=False)
    monkeypatch.chdir(first)

    import ts_agents.config as config

    importlib.reload(config)
    monkeypatch.chdir(second)

    assert config.PERSISTENCE_DIR == second
    assert config.RESULTS_CACHE_DIR == second / "results"
