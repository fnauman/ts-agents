"""Shared data access utilities for UI, tools, and CLI.

Centralizes data loading, caching, variable alias resolution, and series retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Iterable, Union

import pandas as pd

from . import data_loader
from . import config

# Cache dataframes by (data_type, use_test_data, data_dir, test_file)
_DATAFRAME_CACHE: Dict[Tuple[str, bool, str, str], pd.DataFrame] = {}
_METADATA_COLUMNS = {"unique_id", "ds", "time", "t"}


def clear_cache() -> None:
    """Clear cached dataframes (useful for tests)."""
    _DATAFRAME_CACHE.clear()


def _cache_key(data_type: str, use_test_data: bool) -> Tuple[str, bool, str, str]:
    return (
        data_type,
        bool(use_test_data),
        str(config.DATA_DIR),
        str(config.TEST_DATA_FILE),
    )


def infer_data_type(variable_name: str) -> str:
    """Infer data type from variable name or config lists."""
    if "real" in variable_name or variable_name in config.REAL_VARIABLES:
        return "real"
    if "imag" in variable_name or variable_name in config.IMAG_VARIABLES:
        return "imag"
    return "real"


def resolve_variable_name(variable_name: str, df: pd.DataFrame) -> str:
    """Resolve variable aliases using the dataframe columns."""
    if variable_name in config.VARIABLE_ALIASES:
        resolved = config.VARIABLE_ALIASES[variable_name]
        if resolved in df.columns:
            return resolved
        # Handle 'y' alias for imag datasets if present
        if variable_name == "y" and "by001_imag" in df.columns:
            return "by001_imag"
    if variable_name in df.columns:
        return variable_name
    return variable_name


def load_dataframe(
    data_type: str = "real",
    use_test_data: Optional[bool] = None,
) -> pd.DataFrame:
    """Load and cache the dataframe for a given data type."""
    if use_test_data is None:
        use_test_data = config.DEFAULT_USE_TEST_DATA

    key = _cache_key(data_type, use_test_data)
    if key not in _DATAFRAME_CACHE:
        _DATAFRAME_CACHE[key] = data_loader.load_data(
            data_type=data_type,
            use_test_data=use_test_data,
        )
    return _DATAFRAME_CACHE[key]


def get_series(
    run_id: str,
    variable_name: str,
    use_test_data: Optional[bool] = None,
    data_type: Optional[str] = None,
):
    """Get a series for a run/variable with alias resolution."""
    if data_type is None:
        data_type = infer_data_type(variable_name)

    df = load_dataframe(data_type=data_type, use_test_data=use_test_data)
    resolved = resolve_variable_name(variable_name, df)
    return data_loader.get_series(df, run_id, resolved)


def list_runs(
    data_type: str = "real",
    use_test_data: Optional[bool] = None,
) -> List[str]:
    """List available run IDs for a given dataset."""
    df = load_dataframe(data_type=data_type, use_test_data=use_test_data)
    return data_loader.list_runs(df)


def list_variables(
    data_type: str = "real",
    use_test_data: Optional[bool] = None,
    include_aliases: bool = True,
    exclude_columns: Optional[Iterable[str]] = None,
) -> List[str]:
    """List available variable columns for a given dataset.

    Parameters
    ----------
    data_type : str
        "real" or "imag"
    use_test_data : bool, optional
        Whether to use test data
    include_aliases : bool
        Whether to include configured aliases (e.g., "y")
    exclude_columns : Iterable[str], optional
        Column names to exclude (defaults to metadata columns)
    """
    df = load_dataframe(data_type=data_type, use_test_data=use_test_data)
    excluded = set(exclude_columns) if exclude_columns is not None else _METADATA_COLUMNS
    variables = [c for c in df.columns if c not in excluded]

    if include_aliases:
        for alias in config.VARIABLE_ALIASES.keys():
            if alias not in variables:
                variables.append(alias)

    return sorted(variables)


M4_MONTHLY_MINI_PATH = "data/m4_monthly_mini.csv"
M4_MONTHLY_MINI_FREQ = "MS"
M4_MONTHLY_MINI_HORIZON = 18
M4_MONTHLY_MINI_SEASON_LENGTH = 12
_PANEL_SPLITS = ("train", "holdout", "all")


def month_start(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Parse ``value`` as a timezone-naive month-start date, rejecting anything else."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid month-start date {value!r}: {exc}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Invalid month-start date {value!r}.")
    if timestamp.tzinfo is not None or timestamp != timestamp.normalize() or timestamp.day != 1:
        raise ValueError(f"Expected a timezone-naive month-start date such as 2015-06-01, got {value!r}.")
    return timestamp


def resolve_m4_monthly_mini_path() -> Path:
    """Locate the bundled M4 monthly mini-panel in a checkout or installed wheel."""
    from .runtime_paths import resolve_existing_path

    resolved = resolve_existing_path(M4_MONTHLY_MINI_PATH)
    if resolved is None:
        raise FileNotFoundError(f"Bundled dataset not found: {M4_MONTHLY_MINI_PATH}")
    return resolved


def export_m4_monthly_panel(
    split: str = "train",
    train_end: Union[str, pd.Timestamp] = "2015-06-01",
    source_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Return the bundled M4 monthly mini-panel as ``unique_id, ds, y`` rows.

    The source CSV stores a 1-based integer index per series, so the dates are a
    synthetic alignment: each series' last train observation is mapped to
    ``train_end`` and every other row is offset by whole months from there. All
    series therefore share the train end date and, because every series has the
    same holdout length, the holdout end date too, which is what aligned panel
    backtests (``normalize_panel(freq="MS")``) require.

    Parameters
    ----------
    split : str
        ``"train"``, ``"holdout"`` (the next 18 months), or ``"all"``.
    train_end : str or pd.Timestamp
        Month-start date assigned to the final train observation.
    source_path : str or Path, optional
        Override for the source CSV (defaults to the bundled resource).
    """
    if split not in _PANEL_SPLITS:
        raise ValueError(f"split must be one of {', '.join(_PANEL_SPLITS)}; got {split!r}.")
    anchor = month_start(train_end)

    if source_path is None:
        source_path = resolve_m4_monthly_mini_path()
    frame = pd.read_csv(source_path)
    missing = sorted({"unique_id", "split", "ds", "y"}.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")
    if not frame["split"].isin(["train", "holdout"]).all():
        raise ValueError(f"{source_path} has split values other than train/holdout.")

    frame = frame.astype({"unique_id": str})
    pieces: List[pd.DataFrame] = []
    holdout_lengths = set()
    for series_id, group in frame.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        index = group["ds"].astype(int).to_numpy()
        if len(index) > 1 and not (index[1:] - index[:-1] == 1).all():
            raise ValueError(f"Series {series_id} has a non-contiguous integer time index.")
        is_train = (group["split"] == "train").to_numpy()
        if not is_train.any():
            raise ValueError(f"Series {series_id} has no train rows.")
        last_train_position = int(is_train.nonzero()[0][-1])
        if not is_train[: last_train_position + 1].all():
            raise ValueError(f"Series {series_id} has holdout rows before its last train row.")
        holdout_lengths.add(len(index) - last_train_position - 1)

        start = anchor - pd.DateOffset(months=last_train_position)
        dated = pd.DataFrame(
            {
                "unique_id": series_id,
                "ds": pd.date_range(start, periods=len(group), freq=M4_MONTHLY_MINI_FREQ),
                "y": group["y"].astype(float).to_numpy(),
            }
        )
        if split == "train":
            dated = dated.iloc[: last_train_position + 1]
        elif split == "holdout":
            dated = dated.iloc[last_train_position + 1 :]
        pieces.append(dated)

    if len(holdout_lengths) != 1:
        raise ValueError(
            f"{source_path} series have unequal holdout lengths {sorted(holdout_lengths)}; "
            "the exported panel would not be aligned."
        )
    return pd.concat(pieces, ignore_index=True)
