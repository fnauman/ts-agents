"""Input validation shared by univariate analysis functions.

Univariate tools must not silently reinterpret their input: multichannel
arrays are rejected rather than flattened, and non-finite values are rejected
unless a caller explicitly handles them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_univariate_series(
    series: Any,
    *,
    name: str = "series",
    min_length: int = 1,
    allow_nonfinite: bool = False,
) -> np.ndarray:
    """Return ``series`` as a 1D float64 array or raise ``ValueError``.

    Shapes ``(n,)``, ``(n, 1)``, and ``(1, n)`` are accepted. Anything with
    more than one channel is rejected because flattening would interleave
    channels into one meaningless sequence.
    """
    try:
        array = np.asarray(series, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; got shape {tuple(array.shape)}. "
            "Select a single channel (for example with --value-col) instead of "
            "passing multichannel data to a univariate tool."
        )

    if not allow_nonfinite:
        nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        if nonfinite:
            raise ValueError(
                f"{name} contains {nonfinite} non-finite value(s) (NaN/inf). "
                "Remove or impute them before analysis."
            )

    if array.size < min_length:
        raise ValueError(
            f"{name} has {array.size} value(s); at least {min_length} are required."
        )
    return array
