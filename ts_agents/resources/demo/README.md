# Demos

Three self-contained terminal demos, each with a shell script (runnable
directly) and a VHS tape file (for recording GIFs).

| Demo | Script | Tape | Output dir |
|------|--------|------|------------|
| Activity recognition (synthetic) | `run_demo.sh` | `demo.tape` | `outputs/demo/` |
| Activity recognition (WISDM) | `run_demo_wisdm.sh` | `demo_wisdm.tape` | `outputs/demo_wisdm/` |
| Forecasting comparison | `run_demo_forecasting.sh` | `demo_forecasting.tape` | `outputs/demo_forecasting/` |

---

## Activity recognition — synthetic (original demo)

A *30–60 second terminal walkthrough* showing:

1. generating a labeled-stream dataset (synthetic stairs data by default)
2. selecting a window size automatically
3. evaluating a windowed classifier
4. producing a simple plot + confusion matrix

## Quick demo (synthetic, no downloads)

From the repo root:

```bash
export OPENAI_API_KEY=your-key
uv run ts-agents demo window-classification
```

Outputs will be written under:

- `data/demo_labeled_stream.csv`
- `outputs/demo/window_selection.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/eval.json`
- `outputs/demo/confusion_matrix.png`
- `outputs/demo/report.md`

### Scripted fallback (no API key)

```bash
uv run ts-agents demo window-classification --no-llm
```

The scripted path writes the same core artifacts (including `report.md`) without
calling an LLM.

Or run the legacy script:

```bash
bash demo/run_demo.sh
```

---

## Activity recognition — WISDM (real data)

Same window-size selection + evaluation workflow, but using the checked-in
WISDM accelerometer subset (2 subjects, 6 activities, ~33 k rows).

```bash
bash demo/run_demo_wisdm.sh
```

Outputs (under `outputs/demo_wisdm/`):

- `window_selection.json`
- `window_scores.png`
- `eval.json`
- `confusion_matrix.png`

---

## Forecasting comparison

Compares forecasting methods on the MHD shearing-box dataset (`data/short_real.csv`).

```bash
bash demo/run_demo_forecasting.sh
```

Or via the CLI directly:

```bash
uv run ts-agents demo forecasting --no-llm
```

The CLI default method set is `arima,theta` for stable behavior on the tiny
built-in test data. To include ETS, prefer a larger setup:

```bash
uv run ts-agents demo forecasting --full-data --horizon 12 --methods arima,ets,theta --no-llm
```

Outputs (under `outputs/demo_forecasting/`):

- `forecast_comparison.json`
- `forecast_comparison.png`
- `forecasting_report.md`

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
