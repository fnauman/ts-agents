"""Shared CLI input parsing helpers for tools and workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ts_agents import data_access


@dataclass
class SeriesInput:
    """Normalized series input used by workflow commands."""

    series: np.ndarray
    source_type: str
    label: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    input_path: Optional[str] = None
    time_values: Optional[list[Any]] = None
    time_column: Optional[str] = None
    value_column: Optional[str] = None


@dataclass
class LabeledStreamInput:
    """Normalized labeled-stream input used by activity workflows."""

    values: np.ndarray
    labels: np.ndarray
    source_type: str
    label: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    input_path: Optional[str] = None
    time_values: Optional[list[Any]] = None
    time_column: Optional[str] = None
    value_columns: list[str] = field(default_factory=list)
    label_column: str = "label"


def load_json_value(
    *,
    input_json: Optional[str] = None,
    use_stdin: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Load JSON from an inline string, path, or stdin."""
    if input_json and use_stdin:
        raise ValueError("Use at most one of --input-json and --stdin.")

    if use_stdin:
        raw_text = sys.stdin.read()
        source_type = "stdin_json"
    elif input_json is None:
        return None, None
    elif input_json == "-":
        raw_text = sys.stdin.read()
        source_type = "stdin_json"
    else:
        candidate_path = Path(input_json)
        if candidate_path.exists():
            raw_text = candidate_path.read_text()
            source_type = "json_file"
        else:
            raw_text = input_json
            source_type = "inline_json"

    try:
        return json.loads(raw_text), source_type
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc.msg}") from exc


def load_tool_params_from_json(
    *,
    input_json: Optional[str] = None,
    use_stdin: bool = False,
    param_names: list[str],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load tool parameters from JSON input for `tool run`."""
    payload, source_type = load_json_value(
        input_json=input_json,
        use_stdin=use_stdin,
    )
    if payload is None:
        return {}, None

    if isinstance(payload, dict):
        known_keys = [key for key in payload.keys() if key in param_names]
        unknown_keys = [key for key in payload.keys() if key not in param_names]

        if not unknown_keys:
            return payload, source_type

        if len(param_names) == 1 and not known_keys:
            return {param_names[0]: payload}, source_type

        available = ", ".join(param_names)
        unknown = ", ".join(unknown_keys)
        raise ValueError(
            f"Unknown parameter(s) in JSON input: {unknown}. Available: {available}"
        )

    if len(param_names) == 1:
        return {param_names[0]: payload}, source_type

    if "series" in param_names:
        return {"series": payload}, source_type

    raise ValueError(
        "JSON input for tool execution must be an object keyed by parameter name."
    )


def load_series_input(
    *,
    input_path: Optional[str] = None,
    input_json: Optional[str] = None,
    use_stdin: bool = False,
    run_id: Optional[str] = None,
    variable_name: Optional[str] = None,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    use_test_data: Optional[bool] = None,
) -> SeriesInput:
    """Load a series from bundled data, JSON, or a generic file path."""
    bundled_requested = bool(run_id or variable_name)
    source_count = sum(
        1
        for present in (
            bool(input_path),
            bool(input_json),
            bool(use_stdin),
            bundled_requested,
        )
        if present
    )

    if source_count == 0:
        raise ValueError(
            "Provide one input source via --input, --input-json, --stdin, or --run-id/--variable."
        )
    if source_count > 1:
        raise ValueError(
            "Provide exactly one input source: --input, --input-json, --stdin, or --run-id/--variable."
        )

    if bundled_requested:
        if not run_id or not variable_name:
            raise ValueError("Bundled data inputs require both --run-id and --variable.")
        return _load_bundled_series(
            run_id=run_id,
            variable_name=variable_name,
            use_test_data=use_test_data,
        )

    if input_json or use_stdin:
        payload, source_type = load_json_value(
            input_json=input_json,
            use_stdin=use_stdin,
        )
        if source_type is None:
            raise RuntimeError("load_json_value returned no source type unexpectedly")
        return _series_input_from_json_payload(
            payload,
            source_type=source_type,
            input_path=input_json if source_type == "json_file" else None,
            time_col=time_col,
            value_col=value_col,
        )

    if input_path is None:
        raise RuntimeError("load_series_input requires input_path when no other source is set")
    return _load_series_from_path(
        input_path=input_path,
        time_col=time_col,
        value_col=value_col,
    )


def load_labeled_stream_input(
    *,
    input_path: Optional[str] = None,
    input_json: Optional[str] = None,
    use_stdin: bool = False,
    time_col: Optional[str] = None,
    value_cols: Optional[list[str]] = None,
    label_col: str = "label",
) -> LabeledStreamInput:
    """Load a labeled multivariate stream for activity-recognition workflows."""
    source_count = sum(
        1 for present in (bool(input_path), bool(input_json), bool(use_stdin)) if present
    )
    if source_count == 0:
        raise ValueError("Activity-recognition requires one input source via --input, --input-json, or --stdin.")
    if source_count > 1:
        raise ValueError(
            "Provide exactly one activity-recognition input source: --input, --input-json, or --stdin."
        )

    if input_json or use_stdin:
        payload, source_type = load_json_value(
            input_json=input_json,
            use_stdin=use_stdin,
        )
        if source_type is None:
            raise RuntimeError("load_json_value returned no source type unexpectedly")
        dataframe, label, resolved_input_path = _dataframe_from_json_payload(
            payload,
            source_type=source_type,
            input_path=input_json if source_type == "json_file" else None,
        )
        return _labeled_stream_input_from_dataframe(
            dataframe,
            source_type=source_type,
            label=label,
            input_path=resolved_input_path,
            time_col=time_col,
            value_cols=value_cols,
            label_col=label_col,
        )

    if input_path is None:
        raise RuntimeError("load_labeled_stream_input requires input_path when no other source is set")

    dataframe, source_type, label = _load_dataframe_from_path(input_path)
    return _labeled_stream_input_from_dataframe(
        dataframe,
        source_type=source_type,
        label=label,
        input_path=input_path,
        time_col=time_col,
        value_cols=value_cols,
        label_col=label_col,
    )


def _load_bundled_series(
    *,
    run_id: str,
    variable_name: str,
    use_test_data: Optional[bool],
) -> SeriesInput:
    series = np.asarray(
        data_access.get_series(
            run_id=run_id,
            variable_name=variable_name,
            use_test_data=use_test_data,
        ),
        dtype=np.float64,
    ).flatten()
    return SeriesInput(
        series=series,
        source_type="bundled_run",
        label=f"{variable_name} ({run_id})",
        provenance={
            "series_ref": {
                "source_type": "bundled_run",
                "run_id": run_id,
                "variable": variable_name,
            }
        },
    )


def _load_series_from_path(
    *,
    input_path: str,
    time_col: Optional[str],
    value_col: Optional[str],
) -> SeriesInput:
    if input_path == "-":
        raw_text = sys.stdin.read()
        return _load_series_from_text(
            raw_text,
            time_col=time_col,
            value_col=value_col,
        )

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        return _series_input_from_json_payload(
            payload,
            source_type="json_file",
            input_path=str(path),
            time_col=time_col,
            value_col=value_col,
        )

    import pandas as pd

    if suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
        source_type = "json_file"
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=delimiter)
        source_type = "csv"
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        source_type = "parquet"
    else:
        raise ValueError(
            f"Unsupported input format for '{input_path}'. Supported: csv, tsv, parquet, json."
        )

    return _series_input_from_dataframe(
        df,
        source_type=source_type,
        label=path.name,
        input_path=str(path),
        time_col=time_col,
        value_col=value_col,
    )


def _load_series_from_text(
    raw_text: str,
    *,
    time_col: Optional[str],
    value_col: Optional[str],
) -> SeriesInput:
    stripped = raw_text.lstrip()
    if not stripped:
        raise ValueError("No input data was received from stdin.")

    if stripped[0] in "[{":
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            return _series_input_from_json_payload(
                payload,
                source_type="stdin_json",
                input_path="-",
                time_col=time_col,
                value_col=value_col,
            )

    import pandas as pd

    df = pd.read_csv(StringIO(raw_text))
    return _series_input_from_dataframe(
        df,
        source_type="csv",
        label="stdin.csv",
        input_path="-",
        time_col=time_col,
        value_col=value_col,
    )


def _series_input_from_json_payload(
    payload: Any,
    *,
    source_type: str,
    input_path: Optional[str],
    time_col: Optional[str],
    value_col: Optional[str],
) -> SeriesInput:
    if isinstance(payload, dict) and "series" in payload:
        time_values = payload.get(time_col) if time_col and time_col in payload else None
        if time_values is None:
            time_values = payload.get("time") or payload.get("index")
        label = payload.get("name") or payload.get("label") or "series"
        series = _coerce_series(payload["series"])
        if time_values is not None and len(time_values) != len(series):
            raise ValueError("Time values must have the same length as the series.")
        provenance = {
            "series_ref": {
                "source_type": source_type,
                "path": input_path,
            }
        }
        return SeriesInput(
            series=series,
            source_type=source_type,
            label=label,
            provenance=provenance,
            input_path=input_path,
            time_values=list(time_values) if time_values is not None else None,
            time_column=time_col if time_col and time_values is not None else None,
            value_column=value_col or "series",
        )

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        import pandas as pd

        df = pd.DataFrame(payload)
        return _series_input_from_dataframe(
            df,
            source_type=source_type,
            label=input_path or "series.json",
            input_path=input_path,
            time_col=time_col,
            value_col=value_col,
        )

    if isinstance(payload, dict) and _looks_like_dataframe_payload(payload, value_col=value_col):
        import pandas as pd

        df = pd.DataFrame(payload)
        return _series_input_from_dataframe(
            df,
            source_type=source_type,
            label=input_path or "series.json",
            input_path=input_path,
            time_col=time_col,
            value_col=value_col,
        )

    if isinstance(payload, list):
        return SeriesInput(
            series=_coerce_series(payload),
            source_type=source_type,
            label=input_path or "series",
            provenance={
                "series_ref": {
                    "source_type": source_type,
                    "path": input_path,
                }
            },
            input_path=input_path,
            value_column=value_col or "series",
        )

    raise ValueError(
        "JSON series input must be a list, a {'series': [...]} object, or tabular JSON."
    )


def _dataframe_from_json_payload(
    payload: Any,
    *,
    source_type: str,
    input_path: Optional[str],
):
    import pandas as pd

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return pd.DataFrame(payload), input_path or "inline.json", input_path

    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return pd.DataFrame(payload["records"]), payload.get("name") or input_path or "records.json", input_path
        if payload and all(isinstance(value, list) for value in payload.values()):
            return pd.DataFrame(payload), input_path or "table.json", input_path

    raise ValueError(
        "Tabular JSON input must be a list of records, a {'records': [...]} object, or a dict of equal-length columns."
    )


def _series_input_from_dataframe(
    df,
    *,
    source_type: str,
    label: str,
    input_path: Optional[str],
    time_col: Optional[str],
    value_col: Optional[str],
) -> SeriesInput:
    resolved_value_col = _resolve_value_column(df, value_col=value_col, time_col=time_col)
    if time_col and time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found. Available: {list(df.columns)}"
        )

    series = _coerce_series(df[resolved_value_col].tolist())
    time_values = df[time_col].tolist() if time_col else None
    return SeriesInput(
        series=series,
        source_type=source_type,
        label=label,
        provenance={
            "series_ref": {
                "source_type": source_type,
                "path": input_path,
                "time_column": time_col,
                "value_column": resolved_value_col,
            }
        },
        input_path=input_path,
        time_values=time_values,
        time_column=time_col,
        value_column=resolved_value_col,
    )


def _labeled_stream_input_from_dataframe(
    df,
    *,
    source_type: str,
    label: str,
    input_path: Optional[str],
    time_col: Optional[str],
    value_cols: Optional[list[str]],
    label_col: str,
) -> LabeledStreamInput:
    resolved_value_cols = _resolve_value_columns(
        df,
        value_cols=value_cols,
        time_col=time_col,
        label_col=label_col,
    )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Available: {list(df.columns)}"
        )
    if time_col and time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found. Available: {list(df.columns)}"
        )

    labels = np.asarray(df[label_col].tolist(), dtype=object)
    if labels.size == 0:
        raise ValueError("Label column cannot be empty.")
    if np.any(labels == None):  # noqa: E711
        raise ValueError("Label column contains missing values.")

    values = np.asarray(df[resolved_value_cols].to_numpy(), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Activity-recognition input must contain at least one row of numeric values.")

    return LabeledStreamInput(
        values=values,
        labels=labels,
        source_type=source_type,
        label=label,
        provenance={
            "stream_ref": {
                "source_type": source_type,
                "path": input_path,
                "time_column": time_col,
                "value_columns": resolved_value_cols,
                "label_column": label_col,
            }
        },
        input_path=input_path,
        time_values=df[time_col].tolist() if time_col else None,
        time_column=time_col,
        value_columns=resolved_value_cols,
        label_column=label_col,
    )


def _resolve_value_column(df, *, value_col: Optional[str], time_col: Optional[str]) -> str:
    if value_col:
        if value_col not in df.columns:
            raise ValueError(
                f"Value column '{value_col}' not found. Available: {list(df.columns)}"
            )
        return value_col

    if "y" in df.columns and "y" != time_col:
        return "y"

    numeric_columns = [
        column
        for column in df.columns
        if column != time_col and np.issubdtype(df[column].dtype, np.number)
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    remaining_columns = [column for column in df.columns if column != time_col]
    if len(remaining_columns) == 1:
        return remaining_columns[0]

    raise ValueError(
        "Unable to infer value column. Pass --value-col to select the series column explicitly."
    )


def _resolve_value_columns(
    df,
    *,
    value_cols: Optional[list[str]],
    time_col: Optional[str],
    label_col: str,
) -> list[str]:
    if value_cols:
        missing = [column for column in value_cols if column not in df.columns]
        if missing:
            raise ValueError(
                f"Value column(s) not found: {', '.join(missing)}. Available: {list(df.columns)}"
            )
        return value_cols

    excluded = {label_col}
    if time_col:
        excluded.add(time_col)

    numeric_columns = [
        column
        for column in df.columns
        if column not in excluded and np.issubdtype(df[column].dtype, np.number)
    ]
    if numeric_columns:
        return numeric_columns

    raise ValueError(
        "Unable to infer value columns. Pass --value-cols to select one or more numeric columns explicitly."
    )


def _looks_like_dataframe_payload(
    payload: Dict[str, Any],
    *,
    value_col: Optional[str],
) -> bool:
    if value_col and value_col in payload:
        return True
    return bool(payload) and all(
        isinstance(value, list) for value in payload.values()
    )


def _coerce_series(values: Any) -> np.ndarray:
    try:
        series = np.asarray(values, dtype=np.float64).flatten()
    except (TypeError, ValueError) as exc:
        raise ValueError("Series values must be numeric.") from exc

    if series.size == 0:
        raise ValueError("Series input cannot be empty.")
    return series


def _load_dataframe_from_path(input_path: str):
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        dataframe, label, resolved_input_path = _dataframe_from_json_payload(
            payload,
            source_type="json_file",
            input_path=str(path),
        )
        return dataframe, "json_file", label
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True), "json_file", path.name
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=delimiter), "csv", path.name
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path), "parquet", path.name

    raise ValueError(
        f"Unsupported input format for '{input_path}'. Supported: csv, tsv, parquet, json, jsonl."
    )
