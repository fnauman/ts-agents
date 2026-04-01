import io
import sys

import numpy as np
import pandas as pd
import pytest

from ts_agents.cli.input_parsing import (
    _labeled_stream_input_from_dataframe,
    load_labeled_stream_input,
)


def test_load_labeled_stream_input_accepts_csv_stdin(monkeypatch):
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "x,y,z,label\n"
            "0,0,0,idle\n"
            "1,1,1,walk\n"
        ),
    )

    stream_input = load_labeled_stream_input(
        use_stdin=True,
        value_cols=["x", "y", "z"],
        label_col="label",
    )

    assert stream_input.source_type == "csv"
    assert stream_input.label == "stdin.csv"
    assert stream_input.input_path == "-"
    assert stream_input.value_columns == ["x", "y", "z"]
    assert stream_input.labels.tolist() == ["idle", "walk"]
    np.testing.assert_allclose(stream_input.values, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


@pytest.mark.parametrize("missing_label", [None, np.nan, pd.NaT])
def test_labeled_stream_input_rejects_missing_labels(missing_label):
    dataframe = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "label": ["idle", missing_label],
        }
    )

    with pytest.raises(ValueError, match="Label column contains missing values."):
        _labeled_stream_input_from_dataframe(
            dataframe,
            source_type="csv",
            label="stream.csv",
            input_path="stream.csv",
            time_col=None,
            value_cols=["x", "y"],
            label_col="label",
        )
