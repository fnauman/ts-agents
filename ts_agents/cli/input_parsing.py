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
    labels_input_path: Optional[str] = None,
    labels_input_json: Optional[str] = None,
    labels_time_col: Optional[str] = None,
    label_start_col: Optional[str] = None,
    label_end_col: Optional[str] = None,
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

    label_source_count = sum(
        1 for present in (bool(labels_input_path), bool(labels_input_json)) if present
    )
    if label_source_count > 1:
        raise ValueError(
            "Provide at most one separate labels source via --labels-input or --labels-input-json."
        )

    dataframe, source_type, label, resolved_input_path = _load_dataframe_input_source(
        input_path=input_path,
        input_json=input_json,
        use_stdin=use_stdin,
    )

    label_source_ref: Optional[dict[str, Any]] = None
    if label_source_count:
        labels_dataframe, labels_source_type, labels_label, labels_resolved_input_path = _load_dataframe_input_source(
            input_path=labels_input_path,
            input_json=labels_input_json,
            use_stdin=False,
        )
        dataframe, preparation = _merge_labeled_stream_sources(
            signal_df=dataframe,
            labels_df=labels_dataframe,
            time_col=time_col,
            label_col=label_col,
            labels_time_col=labels_time_col,
            label_start_col=label_start_col,
            label_end_col=label_end_col,
        )
        label_source_ref = {
            "source_type": labels_source_type,
            "path": labels_resolved_input_path,
            "label": labels_label,
            "time_column": labels_time_col,
            "start_column": preparation.get("label_start_col"),
            "end_column": preparation.get("label_end_col"),
            "mode": preparation.get("mode"),
        }
    stream_input = _labeled_stream_input_from_dataframe(
        dataframe,
        source_type=source_type,
        label=label,
        input_path=resolved_input_path,
        time_col=time_col,
        value_cols=value_cols,
        label_col=label_col,
    )
    if label_source_ref is not None:
        stream_ref = stream_input.provenance.setdefault("stream_ref", {})
        stream_ref["label_source"] = label_source_ref
        stream_ref["preparation"] = preparation
    return stream_input


def _load_dataframe_input_source(
    *,
    input_path: Optional[str],
    input_json: Optional[str],
    use_stdin: bool,
):
    if use_stdin:
        return _load_dataframe_from_text(
            sys.stdin.read(),
            source_type_json="stdin_json",
            json_input_path="stdin.json",
            tabular_input_path="-",
            csv_label="stdin.csv",
        )

    if input_json:
        payload, source_type = load_json_value(
            input_json=input_json,
            use_stdin=False,
        )
        if source_type is None:
            raise RuntimeError("load_json_value returned no source type unexpectedly")
        dataframe, label, resolved_input_path = _dataframe_from_json_payload(
            payload,
            source_type=source_type,
            input_path=input_json if source_type == "json_file" else None,
        )
        return dataframe, source_type, label, resolved_input_path

    if input_path is None:
        raise RuntimeError("_load_dataframe_input_source requires an input path when no other source is set")

    if input_path == "-":
        return _load_dataframe_from_text(
            sys.stdin.read(),
            source_type_json="stdin_json",
            json_input_path="stdin.json",
            tabular_input_path="-",
            csv_label="stdin.csv",
        )

    dataframe, source_type, label = _load_dataframe_from_path(input_path)
    return dataframe, source_type, label, input_path


def _load_dataframe_from_text(
    raw_text: str,
    *,
    source_type_json: str,
    json_input_path: str,
    tabular_input_path: str,
    csv_label: str,
):
    stripped = raw_text.lstrip()
    if not stripped:
        raise ValueError("No input data was received from stdin.")

    if stripped[0] in "[{":
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            dataframe, label, _ = _dataframe_from_json_payload(
                payload,
                source_type=source_type_json,
                input_path=json_input_path,
            )
            return dataframe, source_type_json, label, tabular_input_path

    import pandas as pd

    dataframe = pd.read_csv(StringIO(raw_text))
    return dataframe, "csv", csv_label, tabular_input_path


def _merge_labeled_stream_sources(
    *,
    signal_df,
    labels_df,
    time_col: Optional[str],
    label_col: str,
    labels_time_col: Optional[str],
    label_start_col: Optional[str],
    label_end_col: Optional[str],
):
    if bool(label_start_col) ^ bool(label_end_col):
        raise ValueError(
            "Provide both --labels-start-col and --labels-end-col when using segment labels."
        )

    auto_start_col = label_start_col
    auto_end_col = label_end_col
    if auto_start_col is None and auto_end_col is None:
        if "start" in labels_df.columns and "end" in labels_df.columns:
            auto_start_col = "start"
            auto_end_col = "end"

    if auto_start_col and auto_end_col:
        return _merge_segment_labels(
            signal_df=signal_df,
            labels_df=labels_df,
            time_col=time_col,
            label_col=label_col,
            label_start_col=auto_start_col,
            label_end_col=auto_end_col,
        )

    if len(labels_df) == len(signal_df):
        return _merge_labels_by_row_alignment(
            signal_df=signal_df,
            labels_df=labels_df,
            label_col=label_col,
        )

    resolved_labels_time_col = labels_time_col
    if resolved_labels_time_col is None and time_col and time_col in labels_df.columns:
        resolved_labels_time_col = time_col
    if time_col and resolved_labels_time_col:
        return _merge_labels_by_time_alignment(
            signal_df=signal_df,
            labels_df=labels_df,
            time_col=time_col,
            labels_time_col=resolved_labels_time_col,
            label_col=label_col,
        )

    raise ValueError(
        "Separate labels input must either align row-for-row with the signal input, "
        "include a matching time column, or provide segment boundaries via "
        "--labels-start-col/--labels-end-col."
    )


def _merge_labels_by_row_alignment(
    *,
    signal_df,
    labels_df,
    label_col: str,
):
    if label_col not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in separate labels input. Available: {list(labels_df.columns)}"
        )

    merged = signal_df.drop(columns=[label_col], errors="ignore").copy()
    merged[label_col] = labels_df[label_col].to_numpy()
    return merged, {
        "mode": "row_aligned_labels",
        "signal_rows": int(len(signal_df)),
        "label_rows": int(len(labels_df)),
    }


def _merge_labels_by_time_alignment(
    *,
    signal_df,
    labels_df,
    time_col: str,
    labels_time_col: str,
    label_col: str,
):
    if time_col not in signal_df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in signal input. Available: {list(signal_df.columns)}"
        )
    if labels_time_col not in labels_df.columns:
        raise ValueError(
            f"Labels time column '{labels_time_col}' not found. Available: {list(labels_df.columns)}"
        )
    if label_col not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in separate labels input. Available: {list(labels_df.columns)}"
        )

    labels_subset = labels_df[[labels_time_col, label_col]].copy()
    if bool(labels_subset[labels_time_col].duplicated().any()):
        raise ValueError(
            f"Separate labels input contains duplicate values in time column '{labels_time_col}'."
        )

    merged = signal_df.drop(columns=[label_col], errors="ignore").merge(
        labels_subset,
        how="left",
        left_on=time_col,
        right_on=labels_time_col,
        validate="m:1",
        sort=False,
    )
    if labels_time_col != time_col and labels_time_col in merged.columns:
        merged = merged.drop(columns=[labels_time_col])

    import pandas as pd

    missing_mask = pd.isna(merged[label_col])
    if bool(np.any(missing_mask)):
        raise ValueError(
            f"Separate labels input did not cover all signal rows; {int(np.sum(missing_mask))} row(s) are unlabeled."
        )

    return merged, {
        "mode": "time_joined_labels",
        "signal_rows": int(len(signal_df)),
        "label_rows": int(len(labels_df)),
        "signal_time_column": time_col,
        "labels_time_column": labels_time_col,
    }


def _merge_segment_labels(
    *,
    signal_df,
    labels_df,
    time_col: Optional[str],
    label_col: str,
    label_start_col: str,
    label_end_col: str,
):
    if label_col not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in separate labels input. Available: {list(labels_df.columns)}"
        )
    for column in (label_start_col, label_end_col):
        if column not in labels_df.columns:
            raise ValueError(
                f"Segment boundary column '{column}' not found in separate labels input. Available: {list(labels_df.columns)}"
            )

    import pandas as pd

    assigned = np.empty(len(signal_df), dtype=object)
    assigned[:] = None
    covered = np.zeros(len(signal_df), dtype=bool)

    if time_col:
        if time_col not in signal_df.columns:
            raise ValueError(
                f"Time column '{time_col}' not found in signal input. Available: {list(signal_df.columns)}"
            )
        signal_axis = signal_df[time_col]
        mode = "segment_time_labels"
    else:
        signal_axis = None
        mode = "segment_index_labels"

    for row_index, row in enumerate(labels_df.itertuples(index=False), start=1):
        start_value = getattr(row, label_start_col)
        end_value = getattr(row, label_end_col)
        label_value = getattr(row, label_col)
        if pd.isna(start_value) or pd.isna(end_value):
            raise ValueError(
                f"Separate labels input contains missing segment boundaries in row {row_index}."
            )
        if pd.isna(label_value):
            raise ValueError(
                f"Separate labels input contains a missing label in row {row_index}."
            )

        if signal_axis is None:
            try:
                start_index = int(start_value)
                end_index = int(end_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Segment boundaries must be integer row offsets when --time-col is not set."
                ) from exc
            if start_index < 0 or end_index > len(signal_df) or end_index <= start_index:
                raise ValueError(
                    f"Invalid segment bounds [{start_index}, {end_index}) for signal length {len(signal_df)}."
                )
            mask = np.zeros(len(signal_df), dtype=bool)
            mask[start_index:end_index] = True
        else:
            try:
                if end_value <= start_value:
                    raise ValueError(
                        f"Invalid segment bounds [{start_value}, {end_value}) in row {row_index}."
                    )
            except TypeError:
                pass
            mask = ((signal_axis >= start_value) & (signal_axis < end_value)).to_numpy(dtype=bool)
            if not bool(np.any(mask)):
                raise ValueError(
                    f"Segment [{start_value}, {end_value}) in row {row_index} did not match any signal rows."
                )

        if bool(np.any(covered & mask)):
            raise ValueError("Separate segment labels overlap on one or more signal rows.")

        assigned[mask] = label_value
        covered[mask] = True

    unlabeled_rows = int(np.sum(~covered))
    if unlabeled_rows:
        raise ValueError(
            f"Separate segment labels left {unlabeled_rows} signal row(s) unlabeled."
        )

    merged = signal_df.drop(columns=[label_col], errors="ignore").copy()
    merged[label_col] = assigned
    return merged, {
        "mode": mode,
        "signal_rows": int(len(signal_df)),
        "label_rows": int(len(labels_df)),
        "signal_time_column": time_col,
        "label_start_col": label_start_col,
        "label_end_col": label_end_col,
        "interval_semantics": "start_inclusive_end_exclusive",
    }


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


def _load_labeled_stream_from_text(
    raw_text: str,
    *,
    time_col: Optional[str],
    value_cols: Optional[list[str]],
    label_col: str,
) -> LabeledStreamInput:
    stripped = raw_text.lstrip()
    if not stripped:
        raise ValueError("No input data was received from stdin.")

    if stripped[0] in "[{":
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            dataframe, label, _ = _dataframe_from_json_payload(
                payload,
                source_type="stdin_json",
                input_path="stdin.json",
            )
            return _labeled_stream_input_from_dataframe(
                dataframe,
                source_type="stdin_json",
                label=label,
                input_path="-",
                time_col=time_col,
                value_cols=value_cols,
                label_col=label_col,
            )

    import pandas as pd

    df = pd.read_csv(StringIO(raw_text))
    return _labeled_stream_input_from_dataframe(
        df,
        source_type="csv",
        label="stdin.csv",
        input_path="-",
        time_col=time_col,
        value_cols=value_cols,
        label_col=label_col,
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
    import pandas as pd

    if bool(np.any(pd.isna(labels))):
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
