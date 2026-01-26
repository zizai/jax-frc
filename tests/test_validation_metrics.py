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


def test_shock_position_finds_discontinuity():
    """shock_position detects steepest gradient location."""
    from jax_frc.validation.metrics import shock_position
    import jax.numpy as jnp

    # Profile with sharp jump at z=0.3
    z = jnp.linspace(0, 1, 100)
    rho = jnp.where(z < 0.3, 1.0, 0.125)

    position = shock_position(rho, z)
    assert abs(position - 0.3) < 0.02  # Within 2 grid cells


def test_shock_position_error_computes_relative():
    """shock_position_error returns relative error vs expected."""
    from jax_frc.validation.metrics import shock_position_error
    import jax.numpy as jnp

    z = jnp.linspace(0, 1, 100)
    rho = jnp.where(z < 0.32, 1.0, 0.125)  # Shock at 0.32

    error = shock_position_error(rho, z, expected=0.3)
    assert abs(error - 0.0667) < 0.01  # (0.32 - 0.3) / 0.3 ≈ 0.067
