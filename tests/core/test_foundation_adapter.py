"""Mocked Darts adapter tests; they run on base installs without darts or torch."""

import json

import numpy as np
import pandas as pd
import pytest

from ts_agents.core.forecasting import catalog, foundation
from ts_agents.workflows.forecast import _is_dependency_failure


class FakeTimeSeries:
    def __init__(self, values):
        self._values = np.asarray(values).reshape(-1, 1)

    @classmethod
    def from_times_and_values(cls, times, values, columns=None):
        assert isinstance(times, pd.RangeIndex) and len(times) == len(values)
        assert columns == ["y"]
        return cls(values)

    def values(self, copy=True):
        return self._values.copy() if copy else self._values

    def __len__(self):
        return len(self._values)


class FakeModel:
    instances: list = []
    fit_error: BaseException | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_lengths = []
        self.predict_dtypes = []
        FakeModel.instances.append(self)

    def fit(self, series):
        if FakeModel.fit_error is not None:
            raise FakeModel.fit_error
        self.fit_lengths.append(len(series))
        return self

    def predict(self, n, series, verbose=None):
        self.predict_dtypes.extend(item.values().dtype for item in series)
        return [FakeTimeSeries(np.full(n, item.values()[-1, 0] + 1.0, dtype=np.float32)) for item in series]


@pytest.fixture
def fake_darts(monkeypatch):
    FakeModel.instances = []
    FakeModel.fit_error = None
    requested = []

    def fake_import_model_class(spec):
        requested.append(spec.darts_class)
        return FakeModel

    monkeypatch.setattr(foundation, "_import_model_class", fake_import_model_class)
    monkeypatch.setattr(foundation, "_timeseries_cls", lambda: FakeTimeSeries)
    monkeypatch.setattr(catalog, "missing_modules", lambda name: [])
    foundation.clear_model_cache()
    yield requested
    foundation.clear_model_cache()


def test_forecast_arrays_constructs_only_requested_pinned_model(fake_darts):
    histories = [np.arange(600.0), np.arange(30.0), np.array([1.0, 2.0])]
    outputs = foundation.forecast_arrays(histories, model="chronos2_small", horizon=12)

    assert fake_darts == ["Chronos2Model"]
    assert len(FakeModel.instances) == 1
    model = FakeModel.instances[0]
    spec = catalog.get_method("chronos2_small").foundation
    assert model.kwargs["hub_model_name"] == spec.hub_model_name
    assert model.kwargs["hub_model_revision"] == spec.hub_model_revision
    assert model.kwargs["input_chunk_length"] == (1, 512)
    assert model.kwargs["output_chunk_length"] == 12
    assert model.kwargs["save_checkpoints"] is False
    assert model.kwargs["pl_trainer_kwargs"]["accelerator"] == "cpu"
    assert model.kwargs["pl_trainer_kwargs"]["enable_checkpointing"] is False
    assert "use_longer_projection_head" not in model.kwargs
    # Fitted once on the longest (truncated) context; every input is float32.
    assert model.fit_lengths == [512]
    assert set(model.predict_dtypes) == {np.dtype(np.float32)}
    assert [output.dtype for output in outputs] == [np.dtype(np.float64)] * 3
    assert [output.shape for output in outputs] == [(12,)] * 3
    assert outputs[1][0] == 30.0


def test_timesfm_long_horizon_enables_longer_projection_head(fake_darts):
    foundation.forecast_arrays([np.arange(400.0)], model="timesfm2p5", horizon=200)
    model = FakeModel.instances[0]
    assert fake_darts == ["TimesFM2p5Model"]
    assert model.kwargs["use_longer_projection_head"] is True
    assert model.kwargs["input_chunk_length"] == (1, 512)


def test_model_cache_reuses_until_cleared(fake_darts):
    foundation.forecast_arrays([np.arange(50.0)], model="chronos2_small", horizon=6)
    foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
    assert len(FakeModel.instances) == 1
    foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=7)
    assert len(FakeModel.instances) == 2
    foundation.clear_model_cache()
    foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
    assert len(FakeModel.instances) == 3


def test_forecast_panel_frame_rebuilds_keyed_future_timestamps(fake_darts):
    history = pd.DataFrame(
        {
            "unique_id": ["b"] * 4 + ["a"] * 4,
            "ds": list(pd.date_range("2024-01-01", periods=4, freq="MS")) * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    frame = foundation.forecast_panel_frame(history, model="patchtst_fm", freq="MS", horizon=2)
    assert list(frame.columns) == ["unique_id", "ds", "patchtst_fm"]
    assert frame.unique_id.tolist() == ["a", "a", "b", "b"]
    assert frame.ds.tolist() == [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-01")] * 2
    assert frame.patchtst_fm.tolist() == [41.0, 41.0, 5.0, 5.0]


def test_forecast_foundation_returns_forecast_result(fake_darts):
    result = foundation.forecast_foundation(np.arange(20.0), model="chronos2", horizon=3, season_length=12)
    assert result.method == "chronos2"
    assert result.horizon == 3
    assert result.lower_bound is None and result.upper_bound is None
    np.testing.assert_allclose(result.forecast, [20.0, 20.0, 20.0])


def test_not_imported_darts_stub_raises_install_hint(monkeypatch):
    import sys
    import types

    class NotImportedModule:
        def __init__(self, *args, **kwargs):
            pass

    darts_pkg = types.ModuleType("darts")
    models = types.ModuleType("darts.models")
    utils = types.ModuleType("darts.utils")
    utils_utils = types.ModuleType("darts.utils.utils")
    utils_utils.NotImportedModule = NotImportedModule
    models.Chronos2Model = NotImportedModule()
    darts_pkg.models = models
    for name, module in {
        "darts": darts_pkg,
        "darts.models": models,
        "darts.utils": utils,
        "darts.utils.utils": utils_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(ImportError, match=r"ts-agents\[foundation\]"):
        foundation._import_model_class(catalog.get_method("chronos2_small").foundation)


def test_missing_modules_raise_import_error_before_loading(monkeypatch):
    monkeypatch.setattr(catalog, "missing_modules", lambda name: ["darts", "huggingface_hub"])
    monkeypatch.setattr(
        foundation, "_load", lambda *args, **kwargs: pytest.fail("must not load without darts")
    )
    with pytest.raises(ImportError, match=r"ts-agents\[foundation\]"):
        foundation.forecast_arrays([np.arange(10.0)], model="chronos2_small", horizon=2)


def test_weights_unavailable_errors_are_not_dependency_failures(fake_darts):
    FakeModel.fit_error = ConnectionError("We couldn't connect to the Hub")
    with pytest.raises(foundation.FoundationModelUnavailableError) as excinfo:
        foundation.forecast_arrays([np.arange(10.0)], model="chronos2_small", horizon=2)
    message = str(excinfo.value)
    assert "autogluon/chronos-2-small@" in message and "HF_HOME" in message
    assert not _is_dependency_failure(error_type=type(excinfo.value).__name__, message=message)
    assert foundation._MODEL_CACHE == {}


def _chained(outer: BaseException, inner: BaseException) -> BaseException:
    outer.__context__ = inner
    return outer


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("Trial exceeded 0.3s wall-clock limit."),
        # A SIGALRM landing while a cache probe handles FileNotFoundError.
        _chained(TimeoutError("Trial exceeded 0.3s wall-clock limit."), FileNotFoundError("probe")),
        InterruptedError("interrupted"),
        PermissionError("read-only HF_HOME"),
        OSError(28, "No space left on device"),
    ],
    ids=["timeout", "timeout-during-probe", "interrupted", "permission", "disk-full"],
)
def test_timeouts_and_other_os_errors_are_not_reported_as_uncached_weights(fake_darts, error):
    FakeModel.fit_error = error
    with pytest.raises(type(error)) as excinfo:
        foundation.forecast_arrays([np.arange(10.0)], model="chronos2_small", horizon=2)
    assert excinfo.value is error
    assert not isinstance(excinfo.value, foundation.FoundationModelUnavailableError)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("config.json not in cache"),
        _chained(RuntimeError("load failed"), ConnectionRefusedError("refused")),
    ],
    ids=["missing-file", "connection-refused"],
)
def test_lookup_failures_are_reported_as_uncached_weights(fake_darts, error):
    FakeModel.fit_error = error
    with pytest.raises(foundation.FoundationModelUnavailableError):
        foundation.forecast_arrays([np.arange(10.0)], model="chronos2_small", horizon=2)


def test_trial_timeout_during_weight_load_stays_a_timeout(fake_darts, monkeypatch):
    import time

    from ts_agents.autoresearch.runner import _trial_timeout

    monkeypatch.setattr(FakeModel, "fit", lambda self, series: time.sleep(2))
    with pytest.raises(TimeoutError):
        with _trial_timeout(0.2):
            foundation.forecast_arrays([np.arange(10.0)], model="timesfm2p5", horizon=2)


def test_unscoped_cache_keeps_at_most_one_model(fake_darts):
    for horizon in (6, 7, 8, 9):
        foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=horizon)
        assert len(foundation._MODEL_CACHE) == 1
    assert list(foundation._MODEL_CACHE) == [("chronos2_small", 512, 9, "cpu", 0)]
    foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=9)
    assert len(FakeModel.instances) == 4


def test_model_cache_scope_keeps_models_until_outermost_exit(fake_darts):
    with foundation.model_cache_scope():
        foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
        with foundation.model_cache_scope():
            foundation.forecast_arrays([np.arange(80.0)], model="timesfm2p5", horizon=6)
            foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=7)
        # The inner exit releases nothing; every model is still reusable.
        assert len(foundation._MODEL_CACHE) == 3
        foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
        assert len(FakeModel.instances) == 3
    assert foundation._MODEL_CACHE == {}


def test_model_cache_scope_releases_models_on_error(fake_darts):
    with pytest.raises(RuntimeError, match="boom"):
        with foundation.model_cache_scope():
            foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
            raise RuntimeError("boom")
    assert foundation._MODEL_CACHE == {}


def test_foundation_agent_tools_release_weights_after_each_call(fake_darts, monkeypatch):
    from ts_agents.tools import agent_tools

    for horizon in (6, 7, 8):
        result = agent_tools.forecast_foundation(np.arange(50.0), horizon=horizon)
        assert len(result.forecast) == horizon
        assert foundation._MODEL_CACHE == {}
    assert len(FakeModel.instances) == 3

    monkeypatch.setattr(agent_tools, "_get_series_data", lambda variable, run: np.arange(40.0))
    payload = agent_tools.forecast_foundation_with_data("bx001_real", "Re200Rm200", horizon=4)
    assert payload.data["foundation_model"]["output_chunk_length"] == 4
    assert foundation._MODEL_CACHE == {}


def test_dropped_models_in_reference_cycles_are_freed(fake_darts):
    import gc
    import weakref

    gc.disable()  # Only the adapter's explicit collection may free the cycle.
    try:
        foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=6)
        first = FakeModel.instances.pop()
        first.trainer = {"model": first}  # Lightning-style model <-> trainer cycle.
        evicted_ref = weakref.ref(first)
        del first
        # Unscoped: loading a second key evicts and frees the first.
        foundation.forecast_arrays([np.arange(80.0)], model="chronos2_small", horizon=7)
        assert evicted_ref() is None

        second = FakeModel.instances.pop()
        second.trainer = {"model": second}
        cleared_ref = weakref.ref(second)
        del second
        foundation.clear_model_cache()
        assert cleared_ref() is None
    finally:
        gc.enable()


def test_weights_unavailable_maps_to_typed_backend_unavailable_error():
    from ts_agents.cli.main import _exception_to_cli_error, _exit_code_for_exception
    from ts_agents.tools.executor import ToolError, ToolErrorCode

    exc = foundation.FoundationModelUnavailableError("weights for x are not cached; pre-populate HF_HOME")
    error = ToolError.from_exception(exc, tool_name="forecast_foundation_with_data")
    assert error.code == ToolErrorCode.BACKEND_UNAVAILABLE
    assert error.recoverable is True
    assert "HF_HOME" in error.hint
    assert error.details["exception_type"] == "FoundationModelUnavailableError"
    assert _exit_code_for_exception(error) == 5
    # Raised directly (not wrapped by the executor) it maps the same way.
    cli_error = _exception_to_cli_error(exc)
    assert (cli_error.code, cli_error.retryable) == ("backend_unavailable", True)
    assert "HF_HOME" in cli_error.hint
    assert _exit_code_for_exception(exc) == 5


def test_other_fit_errors_propagate_unchanged(fake_darts):
    FakeModel.fit_error = RuntimeError("shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        foundation.forecast_arrays([np.arange(10.0)], model="chronos2_small", horizon=2)


def test_model_spec_is_small_json_without_weights():
    record = foundation.model_spec("patchtst_fm", horizon=18, context_length=256)
    payload = json.dumps(record)
    assert len(payload.encode()) < 2048
    assert record["weights_saved"] is False
    assert record["input_chunk_length"] == [1, 256]
    assert record["output_chunk_length"] == 18
    assert record["trained_horizon"] == 64
    assert record["hub_model_revision"] == catalog.get_method("patchtst_fm").foundation.hub_model_revision
    assert foundation.model_spec("timesfm2p5", horizon=200)["model_kwargs"] == {
        "use_longer_projection_head": True
    }


def test_request_validation_bounds():
    assert foundation.validate_request("chronos2_small", horizon=12) == 512
    with pytest.raises(ValueError, match="horizon"):
        foundation.validate_request("chronos2_small", horizon=2000)
    with pytest.raises(ValueError, match="horizon"):
        foundation.validate_request("chronos2_small", horizon=0)
    with pytest.raises(ValueError, match="context_length"):
        foundation.validate_request("chronos2_small", horizon=12, context_length=0)
    with pytest.raises(ValueError, match="context_length"):
        foundation.validate_request("chronos2_small", horizon=24, context_length=8180)
    assert foundation.resolve_context_length("chronos2_small", None, 1000) == 512
    assert foundation.validate_request("chronos2_small", horizon=1000, context_length=7192) == 7192
    with pytest.raises(ValueError, match="not a foundation model"):
        foundation.validate_request("arima", horizon=12)


def test_input_validation_rejects_empty_and_non_finite(fake_darts):
    with pytest.raises(ValueError, match="empty"):
        foundation.forecast_arrays([np.array([])], model="chronos2_small", horizon=2)
    with pytest.raises(ValueError, match="non-finite"):
        foundation.forecast_arrays([np.array([1.0, np.nan])], model="chronos2_small", horizon=2)
    with pytest.raises(ValueError, match="accelerator"):
        foundation.forecast_arrays([np.arange(5.0)], model="chronos2_small", horizon=2, accelerator="tpu")
    assert FakeModel.instances == []
