# Demo Scripts and Legacy Aliases

These shell scripts and VHS tapes record short terminal demos. The synthetic
activity-recognition and forecasting scripts use the current
`ts-agents workflow run ...` surface. Deprecated compatibility aliases are
documented below for older terminal flows.

| Demo | Script | Tape | Output dir |
|------|--------|------|------------|
| Activity recognition (synthetic) | `run_demo.sh` | `demo.tape` | `outputs/demo/` |
| Activity recognition (WISDM) | `run_demo_wisdm.sh` | `demo_wisdm.tape` | `outputs/demo_wisdm/` |
| Forecasting comparison | `run_demo_forecasting.sh` | `demo_forecasting.tape` | `outputs/demo_forecasting/` |

---

## Activity recognition — synthetic workflow

A *30–60 second terminal walkthrough* showing:

1. generating a labeled-stream dataset (synthetic stairs data by default)
2. selecting a window size automatically
3. evaluating a windowed classifier
4. producing plots, JSON outputs, a run manifest, and `report.md`

```bash
bash demo/run_demo.sh
```

Outputs will be written under:

- `data/demo_labeled_stream.csv`
- `outputs/demo/window_selection.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/eval.json`
- `outputs/demo/confusion_matrix.png`
- `outputs/demo/report.md`
- `outputs/demo/run_manifest.json`

## Compatibility alias (synthetic, no downloads)

From the repo root:

```bash
export OPENAI_API_KEY=your-key
uv run ts-agents demo window-classification
```

The deprecated alias writes similar outputs under:

- `data/demo_labeled_stream.csv`
- `outputs/demo/window_selection.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/eval.json`
- `outputs/demo/confusion_matrix.png`
- `outputs/demo/report.md`

### Deprecated alias without an API key

```bash
uv run ts-agents demo window-classification --no-llm
```

The scripted CLI path writes the same core artifacts (including `report.md`)
without calling an LLM.

---

## Activity recognition — WISDM (real data)

Same window-size selection + evaluation workflow, but using the checked-in
WISDM accelerometer subset (2 subjects, 6 activities, ~33 k rows).

```bash
bash demo/run_demo_wisdm.sh
```

Outputs from the legacy shell script (under `outputs/demo_wisdm/`):

- `window_selection.json`
- `window_scores.png`
- `eval.json`
- `confusion_matrix.png`

If you also want `report.md`, use the deprecated CLI alias directly:

```bash
uv run ts-agents demo window-classification --no-llm \
  --no-generate \
  --csv-path data/wisdm_subset.csv \
  --output-dir outputs/demo_wisdm \
  --report-path outputs/demo_wisdm/report.md
```

That compatibility path writes the same core artifacts plus `outputs/demo_wisdm/report.md`.

---

## Forecasting comparison workflow

Compares forecasting methods on the MHD shearing-box dataset (`data/short_real.csv`).

```bash
bash demo/run_demo_forecasting.sh
```

The shell script runs `workflow run forecast-series` and writes artifacts to
`outputs/demo_forecasting/`:

- `forecast_comparison.json`
- `forecast.json`
- `forecast.csv`
- `report.md`
- `run_manifest.json`
- `forecast_comparison.png` (when plotting is available)

Or via the deprecated compatibility alias:

```bash
uv run ts-agents demo forecasting --no-llm
```

The raw CLI alias defaults to `outputs/demo/` for workflow artifacts and
`outputs/demo/forecasting_report.md` for the compatibility report.

The CLI default method set is `arima,theta` for stable behavior on the tiny
built-in test data. To include ETS, prefer a larger setup:

```bash
uv run ts-agents demo forecasting --full-data --horizon 12 --methods arima,ets,theta --no-llm
```

Outputs from the raw compatibility alias (under `outputs/demo/` by default):

- `forecast_comparison.json`
- `forecast.json`
- `forecast.csv`
- `report.md`
- `forecasting_report.md`
- `forecast_comparison.png` (when plotting is available)
---

## Demo tool bundles (for agent runs)

Use these with `ts-agents agent run --tool-bundle ...`:

- `demo`: meta-bundle (windowing + forecasting demos)
- `demo_windowing`: focused activity-recognition/window-size workflow
- `demo_forecasting`: focused forecasting workflow

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

Each tape writes a GIF to `demo/assets/`.
