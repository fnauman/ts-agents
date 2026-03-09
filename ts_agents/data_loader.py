import logging
from typing import Optional
from pathlib import Path

import pandas as pd

from ts_agents import config

def load_data(data_type: str = "real", use_test_data: Optional[bool] = None):
    """
    Load the CFD data from CSV files.

    Args:
        data_type (str): 'real' or 'imag'
        use_test_data (bool): If True, load the smaller test dataset.
            If None, uses DEFAULT_USE_TEST_DATA from config.

    Returns:
        pd.DataFrame: The loaded data
    """
    if use_test_data is None:
        use_test_data = config.DEFAULT_USE_TEST_DATA

    if use_test_data:
        filename = config.TEST_DATA_FILE
    else:
        filename = f'nixtla_allruns_{data_type}.csv'

    path = config.DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file {filename} not found in {config.DATA_DIR}")

    if use_test_data and data_type != "real":
        logging.getLogger(__name__).warning(
            "Test data file is configured for real-valued data only. "
            "If you need imag variables, set TS_AGENTS_USE_TEST_DATA=false "
            "or provide a compatible test file via TS_AGENTS_TEST_DATA_FILE."
        )

    df = pd.read_csv(path)
    
    # Enforce numeric types for data columns
    # We assume 'unique_id' and 'ds' (if present) are metadata, others are data
    cols_to_convert = [c for c in df.columns if c not in ['unique_id', 'ds', 'time', 't']]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def get_series(df, unique_id, variable_name):
    """
    Extract a specific time series from the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        unique_id (str): The unique run ID (Re/Rm)
        variable_name (str): The variable to extract (e.g., 'bx001_real')
        
    Returns:
        np.ndarray: The time series data
    """
    # Filter by unique_id
    subset = df[df['unique_id'] == unique_id]
    
    if subset.empty:
        raise ValueError(f"No data found for unique_id: {unique_id}")
        
    if variable_name not in subset.columns:
        raise ValueError(f"Variable {variable_name} not found in dataset. Available: {subset.columns.tolist()}")
        
    return subset[variable_name].values

def list_runs(df):
    """List all unique run IDs available in the dataset."""
    return df['unique_id'].unique().tolist()
