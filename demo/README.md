# Demo Scripts

These shell scripts and VHS tapes record short terminal demos. They use the
current `ts-agents workflow run ...` and `ts-agents tool run ...` commands.
The `ts-agents demo ...` command, the packaged demo assets, and the `demo`,
`demo_windowing`, and `demo_forecasting` tool bundles are removed in 0.5.0
(unreleased); run the scripts below or the equivalent workflow commands.

| Demo | Script | Tape | Output dir |
|------|--------|------|------------|
| Activity recognition (synthetic) | `run_demo.sh` | `demo.tape` | `outputs/demo/` |
| Activity recognition (WISDM) | `run_demo_wisdm.sh` | `demo_wisdm.tape` | `outputs/demo_wisdm/` |
| Forecasting comparison | `run_demo_forecasting.sh` | `demo_forecasting.tape` | `outputs/demo_forecasting/` |

---

Before running source-checkout demos, install their analysis profile with
`uv sync --frozen --python 3.12 --extra recommended`. Scripts use `uv run --no-sync` to preserve that
profile and the same Python environment for preparation and summaries. No API
key is needed. The WISDM demo uses the maintained `tool run` commands.

## Activity recognition — synthetic workflow

A *30–60 second terminal walkthrough* showing:

1. generating a labeled-stream dataset (synthetic stairs data by default)
2. selecting a window size automatically
3. evaluating a windowed classifier on held-out segments
4. producing plots, JSON outputs, a run manifest, and `report.md`

```bash
bash demo/run_demo.sh
```

Equivalent commands:

```bash
uv run --no-sync python data/make_synthetic_labeled_stream.py \
  --scenario stairs --hz 20 --minutes 4 --seed 1337 \
  --out data/demo_labeled_stream.csv

uv run --no-sync ts-agents workflow run activity-recognition \
  --input data/demo_labeled_stream.csv \
  --label-col label \
  --value-cols x,y,z \
  --metric balanced_accuracy \
  --classifier auto \
  --output-dir outputs/demo \
  --overwrite
```

Outputs will be written under:

- `data/demo_labeled_stream.csv`
- `outputs/demo/window_selection.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/eval.json`
- `outputs/demo/confusion_matrix.png`
- `outputs/demo/report.md`
- `outputs/demo/run_manifest.json`

The synthetic stairs classes are easy to separate, so the workflow reports
`quality_status: degraded` with the `perfect_metrics` flag. The demo shows the
artifacts, not a realistic accuracy.

---

## Activity recognition — WISDM (real data)

Same window-size selection + evaluation, but using the checked-in WISDM
accelerometer subset (2 subjects, 6 activities, ~33 k rows).

```bash
bash demo/run_demo_wisdm.sh
```

The script calls `select_window_size_from_csv` and
`evaluate_windowed_classifier_from_csv`, then draws the plots with
`demo/plot_window_selection.py` and `demo/plot_confusion_matrix.py`. Outputs
(under `outputs/demo_wisdm/`):

- `window_selection.json`
- `window_scores.png`
- `eval.json`
- `confusion_matrix.png`

The script reuses the data and split used for window selection, so its score
is exploratory and can be optimistic. It also does not pass a subject column,
so test windows come from the same two subjects it trains on.
For an untouched leave-subject-out score plus
`report.md` and `run_manifest.json`, use the workflow with `--group-col`:

```bash
uv run --no-sync ts-agents workflow run activity-recognition \
  --input data/wisdm_subset.csv \
  --label-col label \
  --value-cols x,y,z \
  --group-col subject_id \
  --output-dir outputs/demo_wisdm_workflow
```

---

## Forecasting comparison workflow

```bash
bash demo/run_demo_forecasting.sh
```

The shell script runs `workflow run forecast-series` with
`--run-id Re200Rm200 --variable bx001_real --horizon 1 --methods arima,theta`
and writes artifacts to `outputs/demo_forecasting/`:

- `forecast_comparison.json`
- `forecast.json`
- `forecast.csv`
- `report.md`
- `run_manifest.json`
- `forecast_comparison.png` (when plotting is available)

`--run-id/--variable` reads the bundled `data/short_real.csv` smoke fixture,
which has only 5 points per run. The demo shows the workflow's artifacts and
status fields; it says nothing about forecast accuracy. For a longer series,
run the same workflow on a file:

```bash
uv run --no-sync ts-agents workflow run forecast-series \
  --input data/synthetic_signals.csv \
  --time-col t \
  --value-col periodic \
  --horizon 25 \
  --season-length 25 \
  --methods seasonal_naive,theta,arima \
  --output-dir outputs/demo_forecasting_synthetic
```

---

## Building a custom WISDM stream

WISDM is available on the UCI ML Repository under **CC BY 4.0**.

Make a small stream from one subject (downloads a ~295 MB zip once):

```bash
python data/make_demo_labeled_stream_wisdm.py \
  --subject 1600 --device watch --sensor accel \
  --activities walking,jogging,sitting,standing \
  --trim-policy per_class_seconds \
  --per-class-seconds walking=180,jogging=60,sitting=180,standing=180 \
  --out data/demo_labeled_stream.csv
```

## Recording terminal GIFs

All demos can be recorded with [VHS](https://github.com/charmbracelet/vhs):

```bash
vhs demo/demo.tape              # synthetic activity recognition
vhs demo/demo_wisdm.tape        # WISDM activity recognition
vhs demo/demo_forecasting.tape  # forecasting comparison
```

Each tape types `bash demo/run_demo*.sh` and writes a GIF to `demo/assets/`.
The checked-in `demo/assets/demo.gif` and `demo/assets/demo_forecasting.gif`
were recorded before the latest script changes, so their terminal output can
differ from a fresh run. No WISDM GIF is checked in.
