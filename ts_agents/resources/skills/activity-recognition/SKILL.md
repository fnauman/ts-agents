---
name: time-series-activity-recognition
description: >
  End-to-end workflow for labeled-stream activity recognition: prepare or download data,
  run window-size selection, evaluate a windowed classifier, and produce plots + a short report.
  Use when you need a reproducible CLI workflow artifact or evaluation bundle.
compatibility: "Designed for ts-agents CLI. Also usable by coding agents that can run shell commands."
metadata:
  domain: time-series
  tasks: [classification, activity-recognition, windowing, evaluation]
  ts_agents:
    tool_category: classification
    preferred_workflow: activity-recognition
    preferred_tools: [select_window_size_from_csv, evaluate_windowed_classifier_from_csv]
    artifact_checklist: [window_selection.json, eval.json, report.md]
---

# Activity recognition workflow (labeled stream + window-size selection)

## When to use
Use when you need a reproducible end-to-end activity-recognition workflow artifact (dataset, window-size search, evaluation metrics, plots, and short report).

## Goal
Given a CSV with columns:
- `x,y,z` (values) and
- `label` (per-timepoint activity label)

produce:
- best window size (by balanced accuracy, among windows that keep every class)
- a held-out evaluation: metrics + confusion matrix + chance level
- a plot of score vs window size
- a short report (Markdown)

## How the score is computed
1. Whole groups are held out first: subjects/sessions from `--group-col`, or
   whole label segments without it (`result.data.holdout`).
2. Window size is selected with grouped splits on the remaining data only.
3. The reported `score` comes from one fit on that data scored on the
   untouched holdout, so it is not the best of the selection scores.

When the stream concatenates several subjects or sessions, **always** pass
`--group-col` (for example `--group-col subject_id`). Without it, the holdout
mixes subjects and the score overstates cross-subject performance. Windows
never span two groups, and with `--labeling majority` training windows that
overlap test windows are purged. Compare the score with
`details.chance_balanced_accuracy` in `eval.json`; the
`score_at_or_below_chance` flag marks a classifier that learned nothing.

## Minimal artifact checklist
- `<output-dir>/window_selection.json`
- `<output-dir>/eval.json`
- `<output-dir>/window_scores.png`
- `<output-dir>/confusion_matrix.png`
- `<output-dir>/report.md`

## Workflow
1) Prepare a labeled-stream CSV (synthetic default or real data).
2) Run the `activity-recognition` workflow.
3) Save plots and a short markdown report.

## Fast path: synthetic dataset (no downloads)
Run from repo root:

```bash
python data/make_synthetic_labeled_stream.py \
  --scenario gait --seconds 40 --seed 1337 \
  --out data/demo_labeled_stream.csv
ts-agents workflow run activity-recognition \
  --input data/demo_labeled_stream.csv \
  --label-col label \
  --value-cols x,y,z \
  --output-dir outputs/activity-recognition
```

This:
1. generates `data/demo_labeled_stream.csv`
2. selects a window size and evaluates the chosen classifier
3. writes plots into `outputs/activity-recognition/`
4. writes a short report to `outputs/activity-recognition/report.md`

If signals and labels live in separate files, the workflow can prepare the labeled stream directly:

```bash
ts-agents workflow run activity-recognition \
  --input signals.csv \
  --time-col ts \
  --value-cols x,y,z \
  --labels-input segments.csv \
  --labels-start-col start \
  --labels-end-col end \
  --label-col activity \
  --output-dir outputs/activity-recognition
```

## Customize the workflow

```bash
ts-agents workflow run activity-recognition \
  --input data/demo_labeled_stream.csv \
  --label-col label \
  --value-cols x,y,z \
  --window-sizes 32,64,128 \
  --classifier minirocket \
  --metric balanced_accuracy \
  --labeling majority \
  --stride 16 \
  --n-splits 5 \
  --output-dir outputs/activity-recognition
```

Use the lower-level `tool run select_window_size_from_csv` and
`tool run evaluate_windowed_classifier_from_csv` only when you need more manual
control than the workflow surface provides.

## Real data option: WISDM (UCI, CC BY 4.0)
If you want a real-world dataset:

```bash
python data/make_demo_labeled_stream_wisdm.py \
  --subject 1600 --device watch --sensor accel \
  --activities walking,jogging,sitting,standing \
  --trim-policy per_class_seconds \
  --per-class-seconds walking=180,jogging=60,sitting=180,standing=180 \
  --out data/demo_labeled_stream.csv
```

Then rerun:

```bash
ts-agents workflow run activity-recognition \
  --input data/demo_labeled_stream.csv \
  --label-col label \
  --value-cols x,y,z \
  --output-dir outputs/activity-recognition-wisdm
```

The vendored `data/wisdm_subset.csv` has two subjects; evaluate leave-subject-out with:

```bash
ts-agents workflow run activity-recognition \
  --input data/wisdm_subset.csv \
  --label-col label --group-col subject_id \
  --output-dir outputs/activity-recognition-wisdm-subjects
```

## What to say in the demo (15-30 seconds)
- "We have a long labeled sensor stream; window size matters."
- "We search a small set of candidate windows."
- "We pick the best by balanced accuracy (better under class imbalance)."
- "We score once on held-out subjects the selection never saw."

## Guardrails / common failure fixes
- If `n_windows` is very small: reduce `window_size`, reduce `stride`, or increase dataset length.
- If one class dominates: keep `balance=segment_cap` and lower `max_windows_per_segment`.
- If performance is unstable: run multiple seeds and report mean/std (`--seed` also seeds ROCKET kernels).
- `non_monotonic_time` warning: the time column goes backwards (often timestamp
  resets within a subject); rows are used in input order.

## Outputs to include in a blog post or repo README
- 1 plot: `window_scores.png`
- 1 plot: `confusion_matrix.png`
- 1 short table: best window + balanced accuracy
- 1 command block: `ts-agents workflow run activity-recognition --input <CSV> --label-col label --value-cols x,y,z`

## Report generation standard (Quarto PDF)
For polished deliverables, convert workflow outputs into a Quarto report:

1. Create/update `outputs/reports/activity-recognition.qmd` with:
   - dataset summary
   - window-size sweep figure
   - confusion matrix figure
   - key metrics table and conclusions
2. Render to PDF:

```bash
quarto render outputs/reports/activity-recognition.qmd --to pdf
```

Use clear sectioning, professional figure captions, and reproducible command snippets.
