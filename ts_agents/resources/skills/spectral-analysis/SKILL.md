---
name: time-series-spectral-analysis
description: >
  Frequency-domain analysis: power spectral density, dominant frequencies, spectral slopes
  (e.g. turbulence inertial ranges), significance-tested periodicity, and coherence between two
  aligned signals. Use for physics/engineering/DNS signals, vibration, oscillations, or any
  question about frequencies.
compatibility: "ts-agents CLI (`ts-agents`); base install (numpy/scipy)."
metadata:
  domain: time-series
  tasks: [spectral-analysis, psd, frequency, periodicity, coherence, turbulence]
  ts_agents:
    tool_category: spectral
    preferred_tools: [compute_psd, detect_periodicity, compute_coherence]
---

# Spectral analysis

## When to use
Use when the question is about frequencies, oscillations, spectral slopes, or
how two signals relate across frequencies.

## Units first
Spectral results are only meaningful with the sampling rate:
- Pass a regular numeric time column (`--time-col t`). `sample_rate` is then
  derived from the step and frequencies are in cycles per time-column unit
  (`input.series.sample_rate_source: time_column`).
- Or pass `--param sample_rate=<samples per unit>` explicitly.
- Without either, frequencies are in cycles per sample. Say so in your answer.
- Irregular sampling or gaps (see `input.warnings`) invalidate FFT-based
  results; resample or analyze contiguous stretches first.

## Workflow
### 1) Power spectral density
```bash
ts-agents tool run compute_psd --input data.csv --time-col t --value-col value --json
```
Read `dominant_frequency`, `dominant_period`, and `spectral_slope` (a log-log
fit of PSD vs frequency over all positive frequencies). For turbulence, a slope
near −5/3 suggests an inertial range. The fit spans the full band, so report it
as indicative and, if needed, fit a sub-range yourself from `frequencies`/`psd`.
Welch segment length: `--param nperseg=<samples>` (longer gives finer resolution and more noise).

### 2) Is there a significant period?
```bash
ts-agents tool run detect_periodicity --input data.csv --time-col t --value-col value --json
```
A period is reported (`detected: true`) only when:
- its peak beats an AR(1) red-noise background (Bonferroni-corrected `p_value < alpha`),
- the autocorrelation has a prominent peak at the same lag, and
- the record holds at least `min_cycles` (default 3) cycles.

Otherwise `dominant_period` is null and `candidate_period` shows the strongest
peak. Relax with `--param alpha=0.05` or `--param min_cycles=2` only with a
stated reason.

### 3) Coherence between two signals
`compute_coherence` takes two aligned, equal-length series (unequal lengths are
rejected). Pass them as JSON:
```bash
ts-agents tool run compute_coherence --input-json pair.json --param sample_rate=50 --json
```
where `pair.json` is `{"series1": [...], "series2": [...]}`. Report
`mean_coherence`, `max_coherence`, and `dominant_frequency`.

## Worked example (known answer)
`data/synthetic_signals.csv` is sampled at 50 Hz; the `periodic` column is a 2 Hz sine:
```bash
ts-agents tool run detect_periodicity --input data/synthetic_signals.csv --time-col t --value-col periodic --json
```
Expect `dominant_period ≈ 0.5` (seconds) and `dominant_frequency ≈ 2`.

## Output expectations
- Sampling rate used and its source; the frequency units.
- Dominant frequencies/periods with significance (p-value) where applicable.
- Spectral slope with the caveat about the fit range.
- Caveats: gaps/irregular sampling, record length vs lowest resolvable frequency.
