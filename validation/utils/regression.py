"""Regression utilities for validation time-series comparisons."""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(
    errors: np.ndarray,
    block_size: int,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute mean error and bootstrap CI using fixed-size blocks.

    Args:
        errors: 1D array of per-timepoint errors.
        block_size: Number of points per block.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (0.05 -> 95% CI).

    Returns:
        (mean, lower_ci, upper_ci)
    """
    errors = np.asarray(errors)
    if errors.ndim != 1:
        raise ValueError("errors must be 1D")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    n = len(errors)
    n_blocks = max(1, n // block_size)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.random.randint(0, n_blocks, size=n_blocks)
        sample = np.concatenate(
            [errors[j * block_size:(j + 1) * block_size] for j in idx]
        )
        means[i] = float(np.mean(sample))

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(errors)), lo, hi
