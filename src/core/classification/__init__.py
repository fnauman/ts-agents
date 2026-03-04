"""Time series classification methods.

This module provides functions for classifying time series using
state-of-the-art algorithms from the aeon toolkit:

- Distance-based: DTW + K-Nearest Neighbors
- Convolution-based: ROCKET, MiniRocket, MultiRocket
- Hybrid: HIVE-COTE 2 (HC2)

Data Format
-----------
All classifiers expect data in 3D format: (n_samples, n_channels, n_timepoints)
For univariate time series, use n_channels=1.
"""

from .distance_based import (
    knn_classify,
    compute_dtw_distance,
)

from .convolution import (
    rocket_classify,
    transform_rocket,
)

from .hybrid import (
    hivecote_classify,
    compare_classifiers,
)

from .utils import ensure_3d

__all__ = [
    # Distance-based
    "knn_classify",
    "compute_dtw_distance",
    # Convolution-based
    "rocket_classify",
    "transform_rocket",
    # Hybrid
    "hivecote_classify",
    "compare_classifiers",
    # Utils
    "ensure_3d",
]
