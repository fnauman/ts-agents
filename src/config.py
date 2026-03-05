import os
from pathlib import Path

from .runtime_paths import resolve_default_data_dir

# Load environment variables from ~/.env file if it exists
_env_file = Path.home() / ".env"
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

# =============================================================================
# Data Configuration
# =============================================================================

# Data directory - can be overridden via environment variable
# Note: If the default path doesn't exist, a warning will be logged when data is accessed
# Default to repo-local data in source checkouts, or packaged test data in installs.
_default_data_dir = str(resolve_default_data_dir())
_data_dir_env = os.environ.get("TS_AGENTS_DATA_DIR", _default_data_dir)
DATA_DIR = Path(_data_dir_env)

# Validate data directory exists (just a warning, not an error)
if not DATA_DIR.exists():
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.warning(
        f"Data directory does not exist: {DATA_DIR}. "
        f"Set TS_AGENTS_DATA_DIR environment variable to a valid path, "
        f"or data loading may fail."
    )

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

# Test data file (can be overridden via environment variable)
TEST_DATA_FILE = os.environ.get("TS_AGENTS_TEST_DATA_FILE", "short_real.csv")

# Default to test data unless explicitly disabled
DEFAULT_USE_TEST_DATA = os.environ.get("TS_AGENTS_USE_TEST_DATA", "true").lower() == "true"

# =============================================================================
# Agent Configuration
# =============================================================================

# OpenAI model to use
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# =============================================================================
# Persistence Configuration
# =============================================================================

# Base directory for all persistence (can be overridden via environment)
PERSISTENCE_DIR = Path(os.environ.get(
    "TS_AGENTS_PERSISTENCE_DIR",
    Path.cwd()
))

# Results cache configuration
RESULTS_CACHE_DIR = PERSISTENCE_DIR / "results"
RESULTS_CACHE_ENABLED = os.environ.get("TS_AGENTS_CACHE_ENABLED", "true").lower() == "true"
RESULTS_CACHE_MAX_AGE_DAYS = int(os.environ.get("TS_AGENTS_CACHE_MAX_AGE_DAYS", "30"))

# Session store configuration
SESSIONS_DIR = PERSISTENCE_DIR / "sessions"
SESSIONS_MAX_COUNT = int(os.environ.get("TS_AGENTS_SESSIONS_MAX", "100"))
SESSIONS_TIMEOUT_HOURS = int(os.environ.get("TS_AGENTS_SESSIONS_TIMEOUT_HOURS", "168"))  # 7 days

# Experiment log configuration
EXPERIMENTS_DIR = PERSISTENCE_DIR / "experiments"
EXPERIMENTS_MAX_RUNS = int(os.environ.get("TS_AGENTS_EXPERIMENTS_MAX_RUNS", "1000"))


def init_persistence():
    """Initialize the persistence layer with configured settings.

    This function initializes all persistence components (cache, sessions, experiments)
    with the settings defined in this config module.

    Returns
    -------
    tuple
        (ResultsCache, SessionStore, ExperimentLog) instances
    """
    from .persistence import init_cache, init_session_store, init_experiment_log

    cache = init_cache(
        root_dir=RESULTS_CACHE_DIR,
        enabled=RESULTS_CACHE_ENABLED,
        max_age_days=RESULTS_CACHE_MAX_AGE_DAYS if RESULTS_CACHE_MAX_AGE_DAYS > 0 else None,
    )

    session_store = init_session_store(
        root_dir=SESSIONS_DIR,
        max_sessions=SESSIONS_MAX_COUNT,
        session_timeout_hours=SESSIONS_TIMEOUT_HOURS,
    )

    experiment_log = init_experiment_log(
        root_dir=EXPERIMENTS_DIR,
        max_runs=EXPERIMENTS_MAX_RUNS,
    )

    return cache, session_store, experiment_log
