import importlib

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

    import src.config as config
    import src.data_access as data_access

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

    import src.config as config
    from src.runtime_paths import resolve_default_data_dir

    importlib.reload(config)

    assert config.DATA_DIR == resolve_default_data_dir()
    assert config.DATA_DIR != tmp_path / "data"
