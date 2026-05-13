---
name: report-generation
description: >
  Convert ts-agents workflow outputs, run manifests, plots, JSON, CSV files, and
  short Markdown reports into stakeholder-facing Markdown, Quarto, HTML, or PDF
  reports. Use when the user asks for a report, executive summary, research
  appendix, or reusable analysis deliverable from existing ts-agents artifacts.
compatibility: "Designed for ts-agents CLI artifacts. Also usable by coding agents that can read files and run shell commands."
metadata:
  domain: time-series
  tasks: [reporting, artifact-synthesis, quarto, stakeholder-communication]
  ts_agents:
    preferred_tools: []
    artifact_checklist: [run_manifest.json, report.md]
---

# Report generation from ts-agents artifacts

## When to use
Use this skill when a workflow has already produced artifacts, or when the user
wants a polished report as the final deliverable rather than only terminal
output. Prefer this skill after `inspect-series`, `forecast-series`, or
`activity-recognition` runs.

## Goal
Create a concise, reproducible report that links claims to files:

- the command or workflow that produced the result
- key metrics and quality flags from JSON outputs
- selected plots and tables
- limitations and recommended next steps

## Minimal artifact checklist
- A workflow output directory containing `run_manifest.json`, when available
- The workflow `report.md`, when available
- Referenced plot/table artifacts, for example PNG, CSV, or JSON files
- A final `report.md` or `report.qmd`; optionally rendered HTML or PDF

## Workflow
### 1. Inspect the workflow and output directory
Start from a workflow output directory such as `outputs/forecast/<run-id>/`,
`outputs/activity/<run-id>/`, or an explicit output directory supplied by the
user. Read `run_manifest.json` first when it exists, then inspect the files it
references.

If the user has not run a workflow yet, discover the relevant workflow contract:

```bash
uv run ts-agents workflow show inspect-series --json
uv run ts-agents workflow show forecast-series --json
uv run ts-agents workflow show activity-recognition --json
```

### 2. Choose the report format
Use Markdown for quick internal handoff. Use Quarto HTML for shareable reports.
Use Quarto PDF only when the environment has a working Quarto/PDF toolchain.

Recommended default:

```yaml
---
title: "Time-Series Analysis Report"
format:
  html:
    toc: true
    code-fold: true
---
```

### 3. Compose the report
Use this structure unless the user asks for a different one:

- Summary: state the main result in 2-4 bullets.
- Inputs and run metadata: name the source data, workflow, command options,
  output directory, and any quality flags or warnings.
- Findings: include the most important metrics, plots, and tables. Link every
  figure to a real artifact path.
- Limitations: list data, model, dependency, validation, or workflow
  limitations that affect interpretation.
- Next steps: recommend the next analysis or productionization step.

### 4. Render only after checking paths
Before rendering, verify that every referenced image/table exists and is
readable. If Quarto is installed, render the report:

```bash
quarto render outputs/reports/report.qmd
```

If Quarto is not available, deliver Markdown plus the artifact directory path.

## Workflow-specific guidance
### Forecasting
Use `forecast_comparison.json` for metrics and model ranking,
`forecast_comparison.png` for the visual comparison, `forecast.csv` for the
selected forecast, and `report.md` for the baseline narrative.

### Activity recognition
Use `window_selection.json` for candidate-window scores, `eval.json` for final
metrics and confusion matrix values, `window_scores.png` and
`confusion_matrix.png` for figures, and `report.md` for the summary.

### Inspect series
Use `summary.json` and `ledger.json` for diagnostics, `autocorrelation.png` for
lag structure, and `report.md` for the quick interpretation.

## Guardrails
- Do not invent metrics or conclusions that are not present in artifacts.
- Do not paste large arrays into the report; summarize and link to JSON/CSV.
- Preserve run provenance: command options, output directory, run ID, warnings,
  quality flags, and timestamps when available.
- If a workflow status is degraded, explain why before making recommendations.
- For stakeholder reports, prefer one strong figure per section over a gallery
  of every generated artifact.
