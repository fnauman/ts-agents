import os
from pathlib import Path

from .runtime_paths import resolve_default_data_dir

_USER_ENV_LOADED = False


def load_user_env(force: bool = False) -> bool:
    """Load variables from ``~/.env`` once, without overwriting real env vars."""
    global _USER_ENV_LOADED

    if _USER_ENV_LOADED and not force:
        return False

    _env_file = Path.home() / ".env"
    loaded_any = False
    if _env_file.exists():
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                        loaded_any = True

    _USER_ENV_LOADED = True
    return loaded_any

# =============================================================================
# Data Configuration
# =============================================================================

def get_data_dir() -> Path:
    """Resolve the active data directory lazily."""
    load_user_env()
    return Path(os.environ.get("TS_AGENTS_DATA_DIR", str(resolve_default_data_dir())))

# Available variables in the dataset
# Note: In short_real.csv, 'y' appears as 'by001_real'. In full data, 'y' is used.
REAL_VARIABLES = ["bx001_real", "by001_real", "vx001_imag", "vy001_imag", "ex001_imag", "ey001_imag"]
IMAG_VARIABLES = ["bx001_imag", "by001_imag", "vx001_real", "vy001_real", "ex001_real", "ey001_real"]

# Variable name mappings (user-friendly name -> actual column name)
# 'y' is an alias for by001_real/by001_imag
VARIABLE_ALIASES = {
    "y": "by001_real",  # Will be overridden to by001_imag for imag data
}

# Run IDs available in the dataset
AVAILABLE_RUNS = [
    "Re200Rm200",
    "Re175Rm175",
    "Re150Rm150",
    "Re125Rm125",
    "Re105Rm105",
    "Re102_5Rm102_5",
]

def get_test_data_file() -> str:
    """Return the configured test data filename."""
    load_user_env()
    return os.environ.get("TS_AGENTS_TEST_DATA_FILE", "short_real.csv")

def get_default_use_test_data() -> bool:
    """Return whether test data should be used by default."""
    load_user_env()
    return os.environ.get("TS_AGENTS_USE_TEST_DATA", "true").lower() == "true"

_DYNAMIC_ATTRS = {
    "DATA_DIR": get_data_dir,
    "TEST_DATA_FILE": get_test_data_file,
    "DEFAULT_USE_TEST_DATA": get_default_use_test_data,

}


def __getattr__(name: str):
    try:
        return _DYNAMIC_ATTRS[name]()
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
