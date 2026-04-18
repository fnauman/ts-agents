# Legacy Demo Aliases

Packaged helper assets for the deprecated `ts-agents demo ...` compatibility
surface.

Prefer the workflow/tool grammar from the main docs in new automation. This
README exists to document the compatibility aliases that remain for one release
cycle. The repo-root shell scripts from `demo/` are source-checkout-only and
are not bundled here.

| Demo | Primary command | Tape | Output dir |
|------|-----------------|------|------------|
| Activity recognition (synthetic) | `ts-agents demo window-classification --no-llm` | `demo.tape` | `outputs/demo/` |
| Activity recognition (WISDM) | source checkout only | `demo_wisdm.tape` | `outputs/demo_wisdm/` |
| Forecasting comparison | `ts-agents demo forecasting --no-llm` | `demo_forecasting.tape` | `outputs/demo/` |

---

## Activity recognition — synthetic (legacy demo)

A *30–60 second terminal walkthrough* showing:

1. generating a labeled-stream dataset (synthetic stairs data by default)
2. selecting a window size automatically
3. evaluating a windowed classifier
4. producing a simple plot + confusion matrix

## Compatibility alias (synthetic, no downloads)

From an installed package or source checkout:

```bash
export OPENAI_API_KEY=your-key
ts-agents demo window-classification
```

Outputs will be written under:

- `data/demo_labeled_stream.csv` (repo/source checkout) or bundled demo data
- `outputs/demo/window_selection.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/eval.json`
- `outputs/demo/confusion_matrix.png`
- `outputs/demo/report.md`

### Deprecated alias without an API key

```bash
ts-agents demo window-classification --no-llm
```

This scripted CLI path writes the same core artifacts (including `report.md`)
without calling an LLM.

---

## Activity recognition — WISDM (real data)

Same window-size selection + evaluation workflow, but using the source-checkout
WISDM accelerometer subset (2 subjects, 6 activities, ~33 k rows). This dataset
is not bundled into the published wheel.

From the repo root:

```bash
uv run ts-agents demo window-classification --no-llm \
  --no-generate \
  --csv-path data/wisdm_subset.csv \
  --output-dir outputs/demo_wisdm \
  --report-path outputs/demo_wisdm/report.md
```

Outputs (under `outputs/demo_wisdm/`):

- `window_selection.json`
- `window_scores.png`
- `eval.json`
- `confusion_matrix.png`
- `report.md`

---

## Forecasting comparison (legacy demo)

Compares forecasting methods on the MHD shearing-box dataset (`data/short_real.csv`).

```bash
ts-agents demo forecasting --no-llm
```

The CLI default method set is `arima,theta` for stable behavior on the tiny
built-in test data. To include ETS, prefer a larger setup:

```bash
uv run ts-agents demo forecasting --full-data --horizon 12 --methods arima,ets,theta --no-llm
```

Outputs (under `outputs/demo/` by default):

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

WISDM is available on the UCI ML Repository under **CC BY 4.0**. This is a
source-checkout workflow because the repo-root dataset is not bundled into the
published wheel.

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

From a source checkout, the bundled VHS tapes can be recorded with
[VHS](https://github.com/charmbracelet/vhs):

```bash
vhs demo/demo.tape              # synthetic activity recognition
vhs demo/demo_wisdm.tape        # WISDM activity recognition
vhs demo/demo_forecasting.tape  # forecasting comparison
```

Each tape writes a GIF to `demo/assets/`.
