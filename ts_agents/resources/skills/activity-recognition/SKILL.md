---
name: time-series-activity-recognition
description: >
  End-to-end demo workflow for labeled-stream activity recognition: generate or download data,
  run window-size selection, evaluate a windowed classifier, and produce plots + a short report.
  Use when you need a reproducible CLI demo or evaluation artifact.
compatibility: "Designed for ts-agents CLI. Also usable by coding agents that can run shell commands."
metadata:
  domain: time-series
  tasks: [classification, activity-recognition, windowing, demo, evaluation]
  ts_agents:
    tool_category: classification
---

# Activity recognition demo (labeled stream + window-size selection)

## When to use
Use when you need a reproducible end-to-end activity-recognition demo artifact (dataset, window-size search, evaluation metrics, plots, and short report).

## Goal
Given a CSV with columns:
- `x,y,z` (values) and
- `label` (per-timepoint activity label)

produce:
- best window size (by balanced accuracy)
- final evaluation metrics + confusion matrix
- a plot of score vs window size
- a short report (Markdown)

## Minimal artifact checklist
- `data/demo_labeled_stream.csv`
- `outputs/demo/window_selection.json`
- `outputs/demo/eval.json`
- `outputs/demo/window_scores.png`
- `outputs/demo/confusion_matrix.png`
- `demo/report_template.md` filled in (or a short README section)

## Workflow
1) Prepare a labeled-stream CSV (synthetic default or real data).
2) Run window-size selection.
3) Evaluate a windowed classifier at the chosen size.
4) Save plots and a short markdown report.

## Fast path: synthetic dataset (no downloads)
Run from repo root. Default uses the LLM demo (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=your-key
uv run ts-agents demo window-classification
```

This:
1. generates `data/demo_labeled_stream.csv`
2. selects a window size + evaluates via LLM tool calls
3. writes plots into `outputs/demo/`
4. writes a short report to `outputs/demo/report.md`

## Scripted fallback (no API key)
If you’re running inside a coding agent or CLI without an API key:

```bash
uv run ts-agents demo window-classification --no-llm
```

Or run the legacy script:

```bash
bash demo/run_demo.sh
```

## Real data option: WISDM (UCI, CC BY 4.0)
If you want a real-world dataset:

```bash
python scripts/make_demo_labeled_stream_wisdm.py \
  --subject 1600 --device watch --sensor accel \
  --activities walking,jogging,sitting,standing \
  --trim-policy per_class_seconds \
  --per-class-seconds walking=180,jogging=60,sitting=180,standing=180 \
  --out data/demo_labeled_stream.csv
```

Then rerun:

```bash
bash demo/run_demo.sh
```

## What to say in the demo (15–30 seconds)
- “We have a long labeled sensor stream; window size matters.”
- “We search a small set of candidate windows.”
- “We pick the best by balanced accuracy (better under class imbalance).”
- “We evaluate and output confusion matrix + plots for a quick report.”

## Guardrails / common failure fixes
- If `n_windows` is very small: reduce `window_size`, reduce `stride`, or increase dataset length.
- If one class dominates: keep `balance=segment_cap` and lower `max_windows_per_segment`.
- If performance is unstable: run multiple seeds and report mean/std.

## Outputs to include in a blog post or repo README
- 1 plot: `window_scores.png`
- 1 plot: `confusion_matrix.png`
- 1 short table: best window + balanced accuracy
- 1 command block: `bash demo/run_demo.sh`

## Report generation standard (Quarto PDF)
For polished deliverables, convert demo outputs into a Quarto report:

1. Create/update a `.qmd` report with:
   - dataset summary
   - window-size sweep figure
   - confusion matrix figure
   - key metrics table and conclusions
2. Render to PDF:

```bash
quarto render reports/REPORT.qmd --to pdf
```

Use clear sectioning, professional figure captions, and reproducible command snippets.
