"""Results cache for persisting analysis results.

This module provides automatic caching of analysis results to disk,
ensuring no computation is performed twice. Results are cached based
on a hash of the input data and parameters.
"""

import hashlib
import json
import pickle
import logging
from pathlib import Path
from typing import Callable, Any, Optional, Dict, Union, TypeVar
from datetime import datetime
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _hash_array(arr: np.ndarray) -> str:
    """Compute a hash of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _hash_params(params: Dict[str, Any]) -> str:
    """Compute a hash of parameters dictionary."""
    # Sort keys for deterministic hashing
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]


def _serialize_for_json(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    return obj


class ResultsCache:
    """Persistent cache for analysis results.

    Results are stored as pickle files organized by method category.
    An index file tracks all cached results with metadata.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for cached results
    enabled : bool
        Whether caching is enabled (default: True)
    max_age_days : int, optional
        Maximum age of cached results in days (None = no expiration)

    Examples
    --------
    >>> cache = ResultsCache("./results")
    >>> result = cache.get_or_compute(
    ...     run_id="Re200Rm200",
    ...     variable="bx001_real",
    ...     method="stl_decompose",
    ...     params={"period": 150},
    ...     compute_fn=lambda: stl_decompose(series, period=150)
    ... )
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "./results",
        enabled: bool = True,
        max_age_days: Optional[int] = None,
    ):
        self.root = Path(root_dir)
        self.enabled = enabled
        self.max_age_days = max_age_days
        self._index: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_index()

    @property
    def index_file(self) -> Path:
        """Path to the cache index file."""
        return self.root / "cache_index.json"

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text())
                logger.debug(f"Loaded cache index with {len(self._index)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            self.index_file.write_text(json.dumps(self._index, indent=2))
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")

    def _compute_key(
        self,
        run_id: str,
        variable: str,
        method: str,
        params: Dict[str, Any],
        series_hash: Optional[str] = None,
    ) -> str:
        """Generate a unique cache key from parameters.

        The key is based on:
        - run_id and variable (or series_hash for raw series)
        - method name
        - all parameters that affect the result
        """
        key_parts = [
            series_hash if series_hash else f"{run_id}:{variable}",
            method,
            _hash_params(params),
        ]
        key_data = ":".join(key_parts)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cache_path(self, key: str, method: str) -> Path:
        """Get the file path for a cached result."""
        # Organize by method category (first part of method name)
        category = method.split("_")[0] if "_" in method else "misc"
        category_dir = self.root / category
        category_dir.mkdir(exist_ok=True)
        return category_dir / f"{key}.pkl"

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        if self.max_age_days is None:
            return False

        created = datetime.fromisoformat(entry.get("created", "1970-01-01"))
        age_days = (datetime.now() - created).days
        return age_days > self.max_age_days

    def get(
        self,
        run_id: str,
        variable: str,
        method: str,
        params: Dict[str, Any],
        series_hash: Optional[str] = None,
    ) -> Optional[Any]:
        """Retrieve a cached result if available.

        Parameters
        ----------
        run_id : str
            Run identifier (e.g., "Re200Rm200")
        variable : str
            Variable name (e.g., "bx001_real")
        method : str
            Analysis method name
        params : dict
            Method parameters
        series_hash : str, optional
            Hash of input series (for raw series without run_id)

        Returns
        -------
        Any or None
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None

        key = self._compute_key(run_id, variable, method, params, series_hash)

        if key not in self._index:
            logger.debug(f"Cache miss for {method} on {run_id}/{variable}")
            return None

        entry = self._index[key]

        # Check expiration
        if self._is_expired(entry):
            logger.debug(f"Cache expired for {method} on {run_id}/{variable}")
            self._remove(key)
            return None

        # Load result from disk
        cache_path = self.root / entry["path"]
        if not cache_path.exists():
            logger.warning(f"Cache file missing: {cache_path}")
            del self._index[key]
            self._save_index()
            return None

        try:
            result = pickle.loads(cache_path.read_bytes())
            logger.debug(f"Cache hit for {method} on {run_id}/{variable}")
            return result
        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Failed to load cached result: {e}")
            return None

    def put(
        self,
        run_id: str,
        variable: str,
        method: str,
        params: Dict[str, Any],
        result: Any,
        series_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a result in the cache.

        Parameters
        ----------
        run_id : str
            Run identifier
        variable : str
            Variable name
        method : str
            Analysis method name
        params : dict
            Method parameters
        result : Any
            Result to cache (must be picklable)
        series_hash : str, optional
            Hash of input series
        metadata : dict, optional
            Additional metadata to store

        Returns
        -------
        str
            Cache key for the stored result
        """
        if not self.enabled:
            return ""

        key = self._compute_key(run_id, variable, method, params, series_hash)
        cache_path = self._get_cache_path(key, method)

        try:
            cache_path.write_bytes(pickle.dumps(result))
        except (pickle.PickleError, IOError) as e:
            logger.error(f"Failed to cache result: {e}")
            return ""

        self._index[key] = {
            "path": str(cache_path.relative_to(self.root)),
            "run_id": run_id,
            "variable": variable,
            "method": method,
            "params": _serialize_for_json(params),
            "created": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_index()

        logger.debug(f"Cached {method} result for {run_id}/{variable}")
        return key

    def _remove(self, key: str) -> None:
        """Remove a cached result."""
        if key in self._index:
            entry = self._index[key]
            cache_path = self.root / entry["path"]
            if cache_path.exists():
                cache_path.unlink()
            del self._index[key]
            self._save_index()

    def get_or_compute(
        self,
        run_id: str,
        variable: str,
        method: str,
        params: Dict[str, Any],
        compute_fn: Callable[[], T],
        series_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Get cached result or compute and cache it.

        This is the primary method for using the cache. It handles
        checking for existing results and storing new ones automatically.

        Parameters
        ----------
        run_id : str
            Run identifier
        variable : str
            Variable name
        method : str
            Analysis method name
        params : dict
            Method parameters
        compute_fn : callable
            Function to compute the result if not cached
        series_hash : str, optional
            Hash of input series
        metadata : dict, optional
            Additional metadata to store

        Returns
        -------
        T
            The result (cached or newly computed)

        Examples
        --------
        >>> result = cache.get_or_compute(
        ...     run_id="Re200Rm200",
        ...     variable="bx001_real",
        ...     method="stl_decompose",
        ...     params={"period": 150},
        ...     compute_fn=lambda: stl_decompose(series, period=150)
        ... )
        """
        # Try to get from cache
        result = self.get(run_id, variable, method, params, series_hash)
        if result is not None:
            return result

        # Compute and cache
        result = compute_fn()
        self.put(run_id, variable, method, params, result, series_hash, metadata)
        return result

    def list_cached(
        self,
        run_id: Optional[str] = None,
        variable: Optional[str] = None,
        method: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """List cached results with optional filtering.

        Parameters
        ----------
        run_id : str, optional
            Filter by run ID
        variable : str, optional
            Filter by variable
        method : str, optional
            Filter by method

        Returns
        -------
        list of dict
            List of cache entry metadata
        """
        results = []
        for key, entry in self._index.items():
            if run_id and entry.get("run_id") != run_id:
                continue
            if variable and entry.get("variable") != variable:
                continue
            if method and entry.get("method") != method:
                continue
            results.append({"key": key, **entry})
        return results

    def clear(
        self,
        run_id: Optional[str] = None,
        variable: Optional[str] = None,
        method: Optional[str] = None,
    ) -> int:
        """Clear cached results with optional filtering.

        Parameters
        ----------
        run_id : str, optional
            Clear only results for this run
        variable : str, optional
            Clear only results for this variable
        method : str, optional
            Clear only results for this method

        Returns
        -------
        int
            Number of entries cleared
        """
        if run_id is None and variable is None and method is None:
            # Clear all
            count = len(self._index)
            for key in list(self._index.keys()):
                self._remove(key)
            return count

        # Filter and clear
        keys_to_remove = []
        for key, entry in self._index.items():
            if run_id and entry.get("run_id") != run_id:
                continue
            if variable and entry.get("variable") != variable:
                continue
            if method and entry.get("method") != method:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove(key)

        return len(keys_to_remove)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns
        -------
        dict
            Statistics including entry count, size, methods, etc.
        """
        total_size = 0
        methods = set()
        runs = set()

        for entry in self._index.values():
            cache_path = self.root / entry["path"]
            if cache_path.exists():
                total_size += cache_path.stat().st_size
            methods.add(entry.get("method", "unknown"))
            runs.add(entry.get("run_id", "unknown"))

        return {
            "entries": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "methods": sorted(methods),
            "runs": sorted(runs),
            "root_dir": str(self.root),
        }


# Decorator for caching core functions
def cached(
    method_name: str,
    cache: Optional[ResultsCache] = None,
    param_keys: Optional[list[str]] = None,
):
    """Decorator to add caching to a core function.

    Parameters
    ----------
    method_name : str
        Name of the method for cache organization
    cache : ResultsCache, optional
        Cache instance to use (uses default if None)
    param_keys : list of str, optional
        Parameter names to include in cache key (all if None)

    Examples
    --------
    >>> @cached("stl_decompose")
    ... def stl_decompose(series, period=None, robust=True):
    ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            series: np.ndarray,
            *args,
            run_id: str = "unknown",
            variable: str = "unknown",
            _cache: Optional[ResultsCache] = None,
            _skip_cache: bool = False,
            **kwargs,
        ):
            actual_cache = _cache or cache or _get_default_cache()

            if _skip_cache or actual_cache is None or not actual_cache.enabled:
                return func(series, *args, **kwargs)

            # Build params dict for cache key
            params = dict(kwargs)
            if param_keys:
                params = {k: v for k, v in params.items() if k in param_keys}

            # Add positional args to params
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())[1:]  # Skip 'series'
            for i, arg in enumerate(args):
                if i < len(param_names):
                    params[param_names[i]] = arg

            series_hash = _hash_array(series)

            return actual_cache.get_or_compute(
                run_id=run_id,
                variable=variable,
                method=method_name,
                params=params,
                compute_fn=lambda: func(series, *args, **kwargs),
                series_hash=series_hash,
            )

        return wrapper
    return decorator


# Default cache instance
_default_cache: Optional[ResultsCache] = None


def get_default_cache() -> ResultsCache:
    """Get the default cache instance, creating it if needed."""
    global _default_cache
    if _default_cache is None:
        _default_cache = ResultsCache()
    return _default_cache


def _get_default_cache() -> Optional[ResultsCache]:
    """Get the default cache if it exists."""
    return _default_cache


def set_default_cache(cache: ResultsCache) -> None:
    """Set the default cache instance."""
    global _default_cache
    _default_cache = cache


def init_cache(
    root_dir: Union[str, Path] = "./results",
    enabled: bool = True,
    max_age_days: Optional[int] = None,
) -> ResultsCache:
    """Initialize and set the default cache.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for cached results
    enabled : bool
        Whether caching is enabled
    max_age_days : int, optional
        Maximum age of cached results

    Returns
    -------
    ResultsCache
        The initialized cache instance
    """
    cache = ResultsCache(root_dir, enabled, max_age_days)
    set_default_cache(cache)
    return cache
