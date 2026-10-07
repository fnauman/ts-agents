"""Tests for `ts-agents data export-panel` and the M4 panel exporter."""

import json

import pandas as pd
import pytest

from ts_agents import data_access
from ts_agents.cli.input_parsing import load_panel_input
from ts_agents.cli.main import run
from ts_agents.core.forecasting.panel import normalize_panel

SERIES_IDS = ["M10", "M100", "M1000", "M1002", "M4"]


def test_export_train_round_trips_through_panel_loaders(tmp_path, capsys):
    out_path = tmp_path / "nested" / "m4_train.csv"

    code = run(["data", "export-panel", "m4-monthly-mini", "--split", "train", "--out", str(out_path), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "m4-monthly-mini"
    result = payload["result"]
    assert result["path"] == str(out_path.resolve())
    assert result["columns"] == ["unique_id", "ds", "y"]
    assert result["n_series"] == 5
    assert result["series_ids"] == SERIES_IDS
    assert result["end"] == result["train_end"] == "2015-06-01"
    assert result["freq"] == "MS"
    assert result["date_alignment"] == "synthetic"
    assert "--freq MS --horizon 18 --season-length 12" in result["suggested_command"]

    raw = pd.read_csv(out_path)
    assert list(raw.columns) == ["unique_id", "ds", "y"]
    assert len(raw) == result["n_rows"]

    normalized = normalize_panel(raw, "MS")
    assert sorted(normalized.unique_id.unique()) == SERIES_IDS
    ends = normalized.groupby("unique_id").ds.max()
    assert set(ends) == {pd.Timestamp("2015-06-01")}

    panel = load_panel_input(input_path=str(out_path), freq="MS")
    assert len(panel.records) == len(raw)
    assert {record["unique_id"] for record in panel.records} == set(SERIES_IDS)


def test_export_holdout_has_eighteen_aligned_months_per_series(tmp_path):
    out_path = tmp_path / "holdout.csv"

    code = run(["data", "export-panel", "m4-monthly-mini", "--split", "holdout", "--out", str(out_path)])

    assert code == 0
    holdout = normalize_panel(pd.read_csv(out_path), "MS")
    counts = holdout.groupby("unique_id").size()
    assert sorted(counts.index) == SERIES_IDS
    assert set(counts) == {18}
    assert set(holdout.groupby("unique_id").ds.min()) == {pd.Timestamp("2015-07-01")}
    assert set(holdout.groupby("unique_id").ds.max()) == {pd.Timestamp("2016-12-01")}


def test_export_preserves_source_values_and_split_boundary():
    source = pd.read_csv(data_access.resolve_m4_monthly_mini_path())
    full = data_access.export_m4_monthly_panel(split="all", train_end="2010-01-01")
    train = data_access.export_m4_monthly_panel(split="train", train_end="2010-01-01")
    holdout = data_access.export_m4_monthly_panel(split="holdout", train_end="2010-01-01")

    assert len(full) == len(source) == len(train) + len(holdout)
    assert list(full.columns) == ["unique_id", "ds", "y"]
    normalize_panel(full, "MS")
    for series_id, group in source.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        exported = full[full.unique_id == series_id]
        assert exported.y.tolist() == group.y.astype(float).tolist()
        n_train = int((group.split == "train").sum())
        assert train[train.unique_id == series_id].ds.max() == pd.Timestamp("2010-01-01")
        assert exported.ds.iloc[n_train - 1] == pd.Timestamp("2010-01-01")
        assert exported.ds.iloc[n_train] == pd.Timestamp("2010-02-01")


def test_export_rejects_non_month_start_train_end(tmp_path, capsys):
    out_path = tmp_path / "bad.csv"

    code = run(
        ["data", "export-panel", "m4-monthly-mini", "--out", str(out_path), "--train-end", "2015-06-15", "--json"]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "validation_error"
    assert "month-start" in payload["error"]["message"]
    assert not out_path.exists()


def test_export_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be one of"):
        data_access.export_m4_monthly_panel(split="test")


def test_export_rejects_unaligned_holdouts(tmp_path):
    source = tmp_path / "uneven.csv"
    pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "B", "B"],
            "split": ["train", "train", "holdout", "train", "train"],
            "ds": [1, 2, 3, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="unequal holdout lengths"):
        data_access.export_m4_monthly_panel(split="all", source_path=source)
