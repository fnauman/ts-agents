"""Tests for the results cache module."""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.persistence.results_cache import (
    ResultsCache,
    cached,
    init_cache,
    get_default_cache,
    set_default_cache,
)


class TestResultsCache:
    """Tests for ResultsCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance for testing."""
        return ResultsCache(root_dir=temp_cache_dir, enabled=True)

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly."""
        cache = ResultsCache(root_dir=temp_cache_dir)

        assert cache.root.exists()
        assert cache.enabled is True
        # Index file is created on first save, not on initialization
        assert cache._index == {}

    def test_cache_disabled(self, temp_cache_dir):
        """Test cache works when disabled."""
        cache = ResultsCache(root_dir=temp_cache_dir, enabled=False)

        # Put should return empty string
        key = cache.put("run1", "var1", "method1", {}, {"result": 42})
        assert key == ""

        # Get should return None
        result = cache.get("run1", "var1", "method1", {})
        assert result is None

    def test_put_and_get(self, cache):
        """Test storing and retrieving results."""
        result = {"value": 42, "array": np.array([1, 2, 3])}

        cache.put(
            run_id="Re200Rm200",
            variable="bx001_real",
            method="test_method",
            params={"period": 150},
            result=result,
        )

        retrieved = cache.get(
            run_id="Re200Rm200",
            variable="bx001_real",
            method="test_method",
            params={"period": 150},
        )

        assert retrieved is not None
        assert retrieved["value"] == 42
        np.testing.assert_array_equal(retrieved["array"], np.array([1, 2, 3]))

    def test_cache_miss(self, cache):
        """Test cache returns None for missing entries."""
        result = cache.get(
            run_id="nonexistent",
            variable="nonexistent",
            method="nonexistent",
            params={},
        )
        assert result is None

    def test_different_params_different_keys(self, cache):
        """Test that different params produce different cache keys."""
        cache.put("run1", "var1", "method1", {"period": 100}, "result1")
        cache.put("run1", "var1", "method1", {"period": 200}, "result2")

        r1 = cache.get("run1", "var1", "method1", {"period": 100})
        r2 = cache.get("run1", "var1", "method1", {"period": 200})

        assert r1 == "result1"
        assert r2 == "result2"

    def test_get_or_compute_cached(self, cache):
        """Test get_or_compute returns cached result."""
        compute_count = [0]

        def compute():
            compute_count[0] += 1
            return "computed_result"

        # First call should compute
        result1 = cache.get_or_compute(
            run_id="run1",
            variable="var1",
            method="method1",
            params={},
            compute_fn=compute,
        )
        assert result1 == "computed_result"
        assert compute_count[0] == 1

        # Second call should use cache
        result2 = cache.get_or_compute(
            run_id="run1",
            variable="var1",
            method="method1",
            params={},
            compute_fn=compute,
        )
        assert result2 == "computed_result"
        assert compute_count[0] == 1  # Should not have computed again

    def test_list_cached(self, cache):
        """Test listing cached entries."""
        cache.put("run1", "var1", "method1", {}, "r1")
        cache.put("run1", "var2", "method1", {}, "r2")
        cache.put("run2", "var1", "method2", {}, "r3")

        # List all
        all_entries = cache.list_cached()
        assert len(all_entries) == 3

        # Filter by run_id
        run1_entries = cache.list_cached(run_id="run1")
        assert len(run1_entries) == 2

        # Filter by variable
        var1_entries = cache.list_cached(variable="var1")
        assert len(var1_entries) == 2

        # Filter by method
        method1_entries = cache.list_cached(method="method1")
        assert len(method1_entries) == 2

    def test_clear_all(self, cache):
        """Test clearing all cached entries."""
        cache.put("run1", "var1", "method1", {}, "r1")
        cache.put("run2", "var2", "method2", {}, "r2")

        count = cache.clear()
        assert count == 2
        assert len(cache.list_cached()) == 0

    def test_clear_filtered(self, cache):
        """Test clearing with filters."""
        cache.put("run1", "var1", "method1", {}, "r1")
        cache.put("run1", "var2", "method1", {}, "r2")
        cache.put("run2", "var1", "method2", {}, "r3")

        count = cache.clear(run_id="run1")
        assert count == 2
        assert len(cache.list_cached()) == 1

    def test_stats(self, cache):
        """Test cache statistics."""
        cache.put("run1", "var1", "method1", {}, "result1")
        cache.put("run2", "var2", "method2", {}, "result2")

        stats = cache.stats()

        assert stats["entries"] == 2
        assert "method1" in stats["methods"]
        assert "method2" in stats["methods"]
        assert "run1" in stats["runs"]
        assert "run2" in stats["runs"]

    def test_series_hash_based_caching(self, cache):
        """Test caching with series hash instead of run_id/variable."""
        series = np.random.randn(100)
        series_hash = "abc123"

        cache.put(
            run_id="",
            variable="",
            method="test_method",
            params={},
            result="cached_result",
            series_hash=series_hash,
        )

        result = cache.get(
            run_id="",
            variable="",
            method="test_method",
            params={},
            series_hash=series_hash,
        )

        assert result == "cached_result"


class TestCachedDecorator:
    """Tests for the @cached decorator."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_cached_decorator(self, temp_cache_dir):
        """Test that @cached decorator caches results."""
        cache = ResultsCache(root_dir=temp_cache_dir)
        call_count = [0]

        @cached("test_method", cache=cache)
        def my_function(series, period=100):
            call_count[0] += 1
            return {"mean": np.mean(series), "period": period}

        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # First call - should compute
        result1 = my_function(
            series,
            period=100,
            run_id="run1",
            variable="var1",
        )
        assert call_count[0] == 1
        assert result1["mean"] == 3.0

        # Second call with same params - should use cache
        result2 = my_function(
            series,
            period=100,
            run_id="run1",
            variable="var1",
        )
        assert call_count[0] == 1  # No additional call

        # Third call with different params - should compute
        result3 = my_function(
            series,
            period=200,
            run_id="run1",
            variable="var1",
        )
        assert call_count[0] == 2

    def test_cached_decorator_skip_cache(self, temp_cache_dir):
        """Test that _skip_cache bypasses caching."""
        cache = ResultsCache(root_dir=temp_cache_dir)
        call_count = [0]

        @cached("test_method", cache=cache)
        def my_function(series):
            call_count[0] += 1
            return np.mean(series)

        series = np.array([1.0, 2.0, 3.0])

        result1 = my_function(series, run_id="run1", variable="var1")
        assert call_count[0] == 1

        # Skip cache - should compute again
        result2 = my_function(
            series,
            run_id="run1",
            variable="var1",
            _skip_cache=True,
        )
        assert call_count[0] == 2


class TestDefaultCache:
    """Tests for default cache management."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
        # Reset default cache after test
        set_default_cache(None)

    def test_init_cache(self, temp_cache_dir):
        """Test init_cache sets the default cache."""
        cache = init_cache(root_dir=temp_cache_dir)

        default = get_default_cache()
        assert default is cache
        assert default.root == temp_cache_dir

    def test_get_default_cache_creates_new(self):
        """Test get_default_cache creates a cache if none exists."""
        set_default_cache(None)

        default = get_default_cache()
        assert default is not None
        assert isinstance(default, ResultsCache)
