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

# =============================================================================
# Agent Configuration
# =============================================================================

def get_openai_model() -> str:
    """Return the configured OpenAI model name."""
    load_user_env()
    return os.environ.get("OPENAI_MODEL", "gpt-5-mini")

# =============================================================================
# Persistence Configuration
# =============================================================================

def get_persistence_dir() -> Path:
    """Return the base directory for persistence, resolved at access time."""
    load_user_env()
    return Path(os.environ.get("TS_AGENTS_PERSISTENCE_DIR", str(Path.cwd())))

def get_results_cache_dir() -> Path:
    return get_persistence_dir() / "results"


def get_results_cache_enabled() -> bool:
    load_user_env()
    return os.environ.get("TS_AGENTS_CACHE_ENABLED", "true").lower() == "true"


def get_results_cache_max_age_days() -> int:
    load_user_env()
    return int(os.environ.get("TS_AGENTS_CACHE_MAX_AGE_DAYS", "30"))

def get_sessions_dir() -> Path:
    return get_persistence_dir() / "sessions"


def get_sessions_max_count() -> int:
    load_user_env()
    return int(os.environ.get("TS_AGENTS_SESSIONS_MAX", "100"))


def get_sessions_timeout_hours() -> int:
    load_user_env()
    return int(os.environ.get("TS_AGENTS_SESSIONS_TIMEOUT_HOURS", "168"))

def get_experiments_dir() -> Path:
    return get_persistence_dir() / "experiments"


def get_experiments_max_runs() -> int:
    load_user_env()
    return int(os.environ.get("TS_AGENTS_EXPERIMENTS_MAX_RUNS", "1000"))


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

    results_cache_dir = get_results_cache_dir()
    results_cache_enabled = get_results_cache_enabled()
    results_cache_max_age_days = get_results_cache_max_age_days()
    sessions_dir = get_sessions_dir()
    sessions_max_count = get_sessions_max_count()
    sessions_timeout_hours = get_sessions_timeout_hours()
    experiments_dir = get_experiments_dir()
    experiments_max_runs = get_experiments_max_runs()

    cache = init_cache(
        root_dir=results_cache_dir,
        enabled=results_cache_enabled,
        max_age_days=results_cache_max_age_days if results_cache_max_age_days > 0 else None,
    )

    session_store = init_session_store(
        root_dir=sessions_dir,
        max_sessions=sessions_max_count,
        session_timeout_hours=sessions_timeout_hours,
    )

    experiment_log = init_experiment_log(
        root_dir=experiments_dir,
        max_runs=experiments_max_runs,
    )

    return cache, session_store, experiment_log


_DYNAMIC_ATTRS = {
    "DATA_DIR": get_data_dir,
    "TEST_DATA_FILE": get_test_data_file,
    "DEFAULT_USE_TEST_DATA": get_default_use_test_data,
    "OPENAI_MODEL": get_openai_model,
    "PERSISTENCE_DIR": get_persistence_dir,
    "RESULTS_CACHE_DIR": get_results_cache_dir,
    "RESULTS_CACHE_ENABLED": get_results_cache_enabled,
    "RESULTS_CACHE_MAX_AGE_DAYS": get_results_cache_max_age_days,
    "SESSIONS_DIR": get_sessions_dir,
    "SESSIONS_MAX_COUNT": get_sessions_max_count,
    "SESSIONS_TIMEOUT_HOURS": get_sessions_timeout_hours,
    "EXPERIMENTS_DIR": get_experiments_dir,
    "EXPERIMENTS_MAX_RUNS": get_experiments_max_runs,
}


def __getattr__(name: str):
    try:
        return _DYNAMIC_ATTRS[name]()
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
