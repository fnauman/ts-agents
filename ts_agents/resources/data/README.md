# Data

This directory documents both:

- data files bundled into the published `ts-agents` wheel
- additional source-checkout datasets that live at the repo root under `data/`

Bundled wheel data:

- `short_real.csv`
- `m4_monthly_mini.csv`
- `demo_labeled_stream.csv`

## WISDM activity-recognition data

Accelerometer data from the
[WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
(phone, 20 Hz). All variants share the same columns:
`subject_id, timestamp, x, y, z, label`.

The commonly used WISDM subset file is **`data/wisdm_subset.csv`** in a source
checkout. It is **not** bundled into the published wheel. A larger version can
also be generated locally if needed.

| File | Subjects | Rows | Size | Version controlled |
|------|----------|------|------|--------------------|
| **`data/wisdm_subset.csv`** | 2 (1600, 1601) | 33,600 | ~2 MB | source checkout only |
| `wisdm_labeled_stream.csv` | 15 (1600–1615) | 394,313 | ~23 MB | no |

Both files cover the same 6 activities (walking, jogging, stairs, sitting,
standing, clapping) and are generated in the source checkout.

### Generating the data (source checkout only)

```bash
# Checked-in subset: 2 subjects, 140s per activity → ~2 MB
uv run python data/make_wisdm_labeled_stream.py \
  --n-subjects 2 --seconds-per-activity 140 --out data/wisdm_subset.csv

# Optional larger file (not version controlled): 15 subjects, full duration → ~23 MB
uv run python data/make_wisdm_labeled_stream.py
```

The `--seconds-per-activity` flag trims each activity to a contiguous block of
that many seconds per subject. Without it, full segments (~3 min each) are kept.

These helper scripts are not bundled into the published wheel. In a source
checkout, `data/make_wisdm_labeled_stream.py` reuses
`data/make_demo_labeled_stream_wisdm.py`. The raw WISDM data must already be
extracted at `data/raw/wisdm/` (see the demo script's `--dataset-zip` /
`--dataset-dir` flags to download it).

### License

The WISDM dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> Weiss, G. (2019). WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
> \[Dataset\]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59.

## `short_real.csv` — MHD shearing-box simulation

A small time-series dataset from a magnetohydrodynamic (MHD) simulation of the
shearing-box induction equation, computed with the
[SNOOPY](https://ipag.osug.fr/~lesurg/snoopy.html) pseudo-spectral code.

Each row records the time evolution of selected Fourier modes of the magnetic
field. The two value columns in the checked-in file are:

| Column | Physical quantity |
|--------|-------------------|
| `bx001_real` | Re(B_x(0,0,1)) — real part of the (0,0,1) Fourier mode of B_x |
| `by001_real` | Re(B_y(0,0,1)) — real part of the (0,0,1) Fourier mode of B_y |

`unique_id` encodes the run parameters: e.g. `Re200Rm200` means Reynolds number
Re = 200 and magnetic Reynolds number Rm = 200. `ds` is the (integer) time step.

| File | Runs | Rows | Size | Version controlled |
|------|------|------|------|--------------------|
| **`short_real.csv`** | 2 (Re200Rm200, Re175Rm175) | 10 | <1 KB | yes |

This file is used as the default test dataset for the **forecasting demo**
(`uv run ts-agents demo forecasting --no-llm`).

## `m4_monthly_mini.csv` — M4 Monthly workflow subset

A vendored mini-panel from the
[M4 Monthly dataset](https://github.com/Nixtla/datasetsforecast), intended for
the "professional forecasting workflow" spec and future smoke tests.

| File | Series | Rows | Horizon | Version controlled |
|------|--------|------|---------|--------------------|
| **`m4_monthly_mini.csv`** | 5 (`M4`, `M10`, `M100`, `M1000`, `M1002`) | 1,466 | 18-step monthly holdout per series | yes |

Schema:

- `unique_id`: M4 Monthly series identifier
- `split`: `train` or `holdout`
- `ds`: 1-based time index within the full series
- `y`: observed value

This subset is intentionally small enough for deterministic tests. It is useful
for workflow validation, but it is **not** meant to stand in for the full M4
benchmark.

## `demo_labeled_stream.csv`

A small single-subject demo stream (4 activities, ~10 min) used for quick demos
and window-size selection tutorials. In a source checkout, generate or refresh
it with `data/make_demo_labeled_stream_wisdm.py`.
