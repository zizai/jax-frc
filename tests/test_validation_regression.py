import numpy as np

from validation.utils.regression import block_bootstrap_ci


def test_block_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    errors = rng.normal(0.05, 0.01, size=200)
    mean, lo, hi = block_bootstrap_ci(errors, block_size=10, n_boot=200)
    assert lo <= mean <= hi
    assert hi - lo > 0
