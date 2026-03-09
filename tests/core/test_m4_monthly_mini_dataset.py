"""Contract tests for the vendored M4 Monthly mini-panel."""

from pathlib import Path

import pandas as pd


EXPECTED_IDS = {"M4", "M10", "M100", "M1000", "M1002"}
EXPECTED_TRAIN_LENGTHS = {
    "M4": 82,
    "M10": 366,
    "M100": 292,
    "M1000": 330,
    "M1002": 306,
}


def test_m4_monthly_mini_dataset_contract():
    repo_path = Path("data/m4_monthly_mini.csv")
    package_path = Path("ts_agents/resources/data/m4_monthly_mini.csv")

    repo_df = pd.read_csv(repo_path)
    package_df = pd.read_csv(package_path)

    pd.testing.assert_frame_equal(repo_df, package_df)

    assert list(repo_df.columns) == ["unique_id", "split", "ds", "y"]
    assert set(repo_df["unique_id"]) == EXPECTED_IDS
    assert set(repo_df["split"]) == {"train", "holdout"}

    counts = repo_df.groupby(["unique_id", "split"]).size().to_dict()
    for unique_id, train_len in EXPECTED_TRAIN_LENGTHS.items():
        assert counts[(unique_id, "train")] == train_len
        assert counts[(unique_id, "holdout")] == 18

        series_df = repo_df[repo_df["unique_id"] == unique_id].sort_values("ds")
        assert series_df["ds"].tolist() == list(range(1, train_len + 18 + 1))
