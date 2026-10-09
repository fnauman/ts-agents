"""Real (unmocked) Darts foundation-model checks on CPU with chronos2_small only.

Gated because the first run downloads ~112 MB of weights into HF_HOME:

    TS_AGENTS_RUN_FOUNDATION_TESTS=1 uv run --frozen python -m pytest -q tests/core/test_foundation_real.py

With a pre-populated cache, add ``HF_HUB_OFFLINE=1`` to run without network.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("TS_AGENTS_RUN_FOUNDATION_TESTS") != "1":
    pytest.skip(
        "set TS_AGENTS_RUN_FOUNDATION_TESTS=1 to run real foundation-model tests",
        allow_module_level=True,
    )
pytest.importorskip("darts")
pytest.importorskip("torch")

from ts_agents.core.forecasting import foundation  # noqa: E402

MODEL = "chronos2_small"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _series(length: int, *, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return scale * (5 + np.sin(2 * np.pi * t / 12) + 0.01 * t) + rng.normal(0, 0.05 * scale, length)


@pytest.fixture(autouse=True)
def _fresh_cache():
    foundation.clear_model_cache()
    yield
    foundation.clear_model_cache()


def test_r1_short_and_long_series_are_finite_deterministic_and_leave_no_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    histories = [_series(200, seed=0), _series(30, seed=1, scale=100.0), _series(5, seed=2)]

    first = foundation.forecast_arrays(histories, model=MODEL, horizon=12)
    foundation.clear_model_cache()
    second = foundation.forecast_arrays(histories, model=MODEL, horizon=12)

    for forecast in first:
        assert forecast.shape == (12,)
        assert forecast.dtype == np.float64
        assert np.isfinite(forecast).all()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not (tmp_path / "darts_logs").exists()
    assert not list(tmp_path.rglob("*.ckpt"))


def test_r2_cached_model_matches_fresh_model_on_new_series():
    series_a = _series(200, seed=3)
    series_b = _series(60, seed=4, scale=10.0)

    foundation.forecast_arrays([series_a], model=MODEL, horizon=12)
    cached = foundation.forecast_arrays([series_b], model=MODEL, horizon=12)[0]
    foundation.clear_model_cache()
    fresh = foundation.forecast_arrays([series_b], model=MODEL, horizon=12)[0]

    np.testing.assert_array_equal(cached, fresh)


def test_r3_only_the_requested_checkpoint_enters_the_hf_cache():
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    def repo_ids() -> set[str]:
        try:
            return {repo.repo_id for repo in scan_cache_dir().repos}
        except CacheNotFound:
            return set()

    before = repo_ids()
    foundation.forecast_foundation(_series(48, seed=5), model=MODEL, horizon=6)
    assert repo_ids() - before <= {"autogluon/chronos-2-small"}


def test_r4_offline_with_empty_cache_raises_unavailable_error(tmp_path):
    code = (
        "import numpy as np\n"
        "from ts_agents.core.forecasting import foundation\n"
        "try:\n"
        "    foundation.forecast_arrays([np.arange(50.0)], model='chronos2_small', horizon=6)\n"
        "except foundation.FoundationModelUnavailableError as exc:\n"
        "    print('UNAVAILABLE', exc)\n"
        "else:\n"
        "    raise SystemExit('expected FoundationModelUnavailableError')\n"
    )
    env = dict(os.environ, HF_HOME=str(tmp_path / "hf"), HF_HUB_OFFLINE="1")
    env.pop("HF_HUB_CACHE", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "UNAVAILABLE weights for autogluon/chronos-2-small@" in completed.stdout


def test_r5_series_validation_and_future_load_one_checkpoint(tmp_path, monkeypatch):
    from ts_agents.cli.input_parsing import SeriesInput
    from ts_agents.workflows.forecast import run_forecast_series_workflow
    original = foundation._import_model_class
    loads = []
    def counted(spec):
        loads.append(spec.hub_model_name)
        return original(spec)
    monkeypatch.setattr(foundation, "_import_model_class", counted)
    result = run_forecast_series_workflow(
        SeriesInput(_series(48, seed=6), "inline_json", "real"),
        output_dir=str(tmp_path), horizon=6, validation_size=4,
        methods=[MODEL], skip_plots=True)
    assert loads == ["autogluon/chronos-2-small"]
    spec = result.data["foundation_models"][MODEL]
    assert spec["output_chunk_length"] == spec["validation_spec"]["output_chunk_length"] == 6
    assert spec["forecast_horizon"] == 6 and spec["validation_spec"]["forecast_horizon"] == 4
    assert len(result.data["forecast"]) == 6
    assert foundation._MODEL_CACHE == {}
