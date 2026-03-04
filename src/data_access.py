"""Shared data access utilities for UI, tools, and CLI.

Centralizes data loading, caching, variable alias resolution, and series retrieval.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, List, Iterable

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
