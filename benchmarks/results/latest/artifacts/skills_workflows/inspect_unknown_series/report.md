### Report on Inspect-Series Workflow

- **Source**: `series`
- **Length**: 20
- **Mean**: 10.5416
- **Std**: 5.7476
- **Min / Max**: 1.1199 / 19.8640
- **Dominant Period**: 20.0000
- **Periodicity Confidence**: 0.6240
- **Top Periods**: 20.0000, 10.0000, 6.6667

#### Recommended Next Steps
- Run forecast-series with a horizon aligned to the detected period (~20.0).
- Inspect decomposition or changepoints because the series shows strong short-lag dependence.
- Treat periodicity findings as low-confidence because the series is short.