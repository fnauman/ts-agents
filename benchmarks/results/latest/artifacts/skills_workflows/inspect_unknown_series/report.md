### Report on Inspect-Series Workflow

#### Data
- **Source**: `series`
- **Length**: 20 (non-finite dropped: 0)
- **Sampling**: no time column (sample index)
- **Mean / Std**: 10.54 / 5.748
- **Min / Max**: 1.12 / 19.86
- **Quality Flags**: none
- **Warnings**: none

#### Structure
- **Significant Period (samples)**: n/a
- **Periodicity p-value**: 0
- **Strongest Candidate (samples)**: 8
- **Significant Periods**: none
- **Lag-1 Autocorrelation**: 0.8529
- **Regime Screen**: [] (changepoint_pelt)

#### Recommended Next Steps
- No significant periodicity: use find_discords with an event-length m for anomalies, and non-seasonal baselines if forecasting.
- Strong short-lag dependence: residual-based anomaly checks should use decomposition residuals.
- The series is short; treat all diagnostics as low-confidence.

#### If Forecasting
- **Start Method**: theta
- **Recommendation Confidence**: 0.4353
- Lag-1 autocorrelation is 0.85, indicating meaningful short-range dependence.
- A low-cost autoregressive-style baseline is a good next step after diagnostics.