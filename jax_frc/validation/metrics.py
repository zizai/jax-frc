"""Built-in validation metrics."""
import jax.numpy as jnp
from typing import Union


def l2_error(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute relative L2 norm error.
    Returns ||actual - expected||_2 / ||expected||_2
    """
    diff = actual - expected
    return float(jnp.linalg.norm(diff) / jnp.linalg.norm(expected))


def linf_error(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute max pointwise error.
    Returns max|actual - expected|
    """
    return float(jnp.max(jnp.abs(actual - expected)))


def rmse_curve(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute root mean square error between curves.
    Returns sqrt(mean((actual - expected)^2))
    """
    diff = actual - expected
    return float(jnp.sqrt(jnp.mean(diff**2)))


def conservation_drift(initial: float, final: float) -> float:
    """Compute relative change in conserved quantity.
    Returns |final - initial| / |initial|
    """
    return float(jnp.abs(final - initial) / jnp.abs(initial))


def shock_position(profile: jnp.ndarray, axis: jnp.ndarray) -> float:
    """Find position of steepest gradient (shock location).

    Args:
        profile: 1D array of field values (e.g., density)
        axis: 1D array of positions along which to search

    Returns:
        Position of maximum |gradient|
    """
    grad = jnp.abs(jnp.diff(profile))
    idx = jnp.argmax(grad)
    # Interpolate to midpoint between cells
    return float((axis[idx] + axis[idx + 1]) / 2)


def shock_position_error(
    profile: jnp.ndarray,
    axis: jnp.ndarray,
    expected: float
) -> float:
    """Compute relative error in shock position.

    Returns: |detected - expected| / |expected|
    """
    detected = shock_position(profile, axis)
    return float(jnp.abs(detected - expected) / jnp.abs(expected))


def reconnection_rate(psi: jnp.ndarray, times: jnp.ndarray) -> float:
    """Compute average reconnection rate dpsi/dt.

    Args:
        psi: Array of reconnected flux values at each time
        times: Array of time values

    Returns:
        Average reconnection rate
    """
    dpsi = jnp.diff(psi)
    dt = jnp.diff(times)
    return float(jnp.mean(dpsi / dt))


def peak_reconnection_rate(
    psi: jnp.ndarray,
    times: jnp.ndarray
) -> tuple[float, float]:
    """Find peak reconnection rate and time.

    Args:
        psi: Array of reconnected flux values at each time
        times: Array of time values

    Returns:
        (peak_rate, time_of_peak)
    """
    dpsi = jnp.diff(psi)
    dt = jnp.diff(times)
    rates = dpsi / dt
    idx = jnp.argmax(rates)
    peak = float(rates[idx])
    t_peak = float((times[idx] + times[idx + 1]) / 2)
    return peak, t_peak


def current_layer_thickness(J: jnp.ndarray, axis: jnp.ndarray) -> float:
    """Compute FWHM of current density profile.

    Args:
        J: 1D current density profile
        axis: Position array

    Returns:
        Full width at half maximum
    """
    J_max = jnp.max(J)
    half_max = J_max / 2
    above_half = J >= half_max
    indices = jnp.where(above_half)[0]
    if len(indices) < 2:
        return 0.0
    return float(axis[indices[-1]] - axis[indices[0]])


def check_tolerance(
    value: float,
    expected: float,
    tolerance: Union[str, float]
) -> dict:
    """Check if value is within tolerance of expected.

    Args:
        value: Computed metric value
        expected: Expected value
        tolerance: Either percentage string like "10%" or absolute float

    Returns:
        dict with 'pass', 'value', 'expected', 'tolerance', 'message'
    """
    if isinstance(tolerance, str) and tolerance.endswith('%'):
        pct = float(tolerance.rstrip('%'))
        threshold = abs(expected) * pct / 100.0
        tol_str = tolerance
    else:
        threshold = float(tolerance)
        tol_str = str(tolerance)

    diff = abs(value - expected)
    passed = diff <= threshold

    return {
        'pass': passed,
        'value': value,
        'expected': expected,
        'tolerance': tol_str,
        'message': '' if passed else f"Value {value} differs from {expected} by {diff:.4g} (tolerance: {tol_str})"
    }


METRIC_FUNCTIONS = {
    'l2_error': l2_error,
    'linf_error': linf_error,
    'rmse_curve': rmse_curve,
    'conservation_drift': conservation_drift,
    'shock_position': shock_position,
    'shock_position_error': shock_position_error,
    'reconnection_rate': reconnection_rate,
    'peak_reconnection_rate': peak_reconnection_rate,
    'current_layer_thickness': current_layer_thickness,
}
