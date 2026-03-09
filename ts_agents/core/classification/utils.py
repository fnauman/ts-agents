"""Shared utilities for classification modules."""

import numpy as np


def ensure_3d(X: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (n_samples, n_channels, n_timepoints).

    Parameters
    ----------
    X : np.ndarray
        Input array. Can be 1D (single univariate sample),
        2D (n_samples, n_timepoints), or 3D (already correct).

    Returns
    -------
    np.ndarray
        3D array with shape (n_samples, n_channels, n_timepoints)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, 1, -1)
    elif X.ndim == 2:
        return X.reshape(X.shape[0], 1, X.shape[1])
    return X
