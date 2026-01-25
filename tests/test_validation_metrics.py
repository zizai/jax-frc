"""Tests for validation metrics."""
import pytest
import jax.numpy as jnp

def test_l2_error_computes_relative_norm():
    """l2_error returns relative L2 norm."""
    from jax_frc.validation.metrics import l2_error

    actual = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([1.1, 2.0, 2.9])

    error = l2_error(actual, expected)
    assert error > 0
    assert error < 0.1  # Small relative error

def test_linf_error_computes_max_error():
    """linf_error returns max pointwise error."""
    from jax_frc.validation.metrics import linf_error

    actual = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([1.0, 2.5, 3.0])

    error = linf_error(actual, expected)
    assert error == pytest.approx(0.5)

def test_rmse_curve_computes_root_mean_square():
    """rmse_curve returns RMSE between curves."""
    from jax_frc.validation.metrics import rmse_curve

    actual = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected = jnp.array([1.0, 2.0, 3.0, 4.0])

    rmse = rmse_curve(actual, expected)
    assert rmse == pytest.approx(0.0)

def test_check_tolerance_percentage():
    """check_tolerance handles percentage tolerances."""
    from jax_frc.validation.metrics import check_tolerance

    result = check_tolerance(value=105.0, expected=100.0, tolerance="10%")
    assert result['pass'] is True

    result = check_tolerance(value=120.0, expected=100.0, tolerance="10%")
    assert result['pass'] is False

def test_check_tolerance_absolute():
    """check_tolerance handles absolute tolerances."""
    from jax_frc.validation.metrics import check_tolerance

    result = check_tolerance(value=1.05, expected=1.0, tolerance=0.1)
    assert result['pass'] is True
