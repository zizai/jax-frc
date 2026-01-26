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


def test_reconnection_rate_from_flux():
    """reconnection_rate computes dpsi/dt."""
    from jax_frc.validation.metrics import reconnection_rate
    import jax.numpy as jnp

    # Linear increase in reconnected flux
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    psi = jnp.array([0.0, 0.1, 0.2, 0.3])

    rate = reconnection_rate(psi, times)
    assert abs(rate - 0.1) < 0.01


def test_peak_reconnection_rate():
    """peak_reconnection_rate finds maximum rate."""
    from jax_frc.validation.metrics import peak_reconnection_rate
    import jax.numpy as jnp

    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    psi = jnp.array([0.0, 0.05, 0.2, 0.25, 0.26])  # Peak rate between t=1 and t=2

    peak, t_peak = peak_reconnection_rate(psi, times)
    assert abs(peak - 0.15) < 0.02
    assert abs(t_peak - 1.5) < 0.5


def test_current_layer_thickness():
    """current_layer_thickness computes FWHM."""
    from jax_frc.validation.metrics import current_layer_thickness
    import jax.numpy as jnp

    # Gaussian current profile
    z = jnp.linspace(-2, 2, 100)
    sigma = 0.5
    J = jnp.exp(-z**2 / (2 * sigma**2))

    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    fwhm = current_layer_thickness(J, z)
    expected_fwhm = 2.355 * sigma
    assert abs(fwhm - expected_fwhm) < 0.1
