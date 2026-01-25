"""Input validation utilities for plasma simulations.

These functions provide runtime validation of simulation parameters
to catch common errors early and provide helpful error messages.
"""

from typing import Any, Tuple, Union
import jax.numpy as jnp

Array = jnp.ndarray


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is strictly positive.

    Args:
        value: The value to check
        name: Parameter name for error messages

    Raises:
        ValidationError: If value <= 0
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: The value to check
        name: Parameter name for error messages

    Raises:
        ValidationError: If value < 0
    """
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_shape(array: Array, expected_shape: Tuple[int, ...], name: str) -> None:
    """Validate that an array has the expected shape.

    Args:
        array: The array to check
        expected_shape: Expected shape tuple
        name: Array name for error messages

    Raises:
        ValidationError: If shape doesn't match
    """
    if array.shape != expected_shape:
        raise ValidationError(
            f"{name} has wrong shape: expected {expected_shape}, got {array.shape}"
        )


def validate_shape_match(
    array1: Array, array2: Array, name1: str, name2: str
) -> None:
    """Validate that two arrays have matching shapes.

    Args:
        array1, array2: Arrays to compare
        name1, name2: Array names for error messages

    Raises:
        ValidationError: If shapes don't match
    """
    if array1.shape != array2.shape:
        raise ValidationError(
            f"Shape mismatch: {name1} has shape {array1.shape}, "
            f"{name2} has shape {array2.shape}"
        )


def validate_finite(array: Array, name: str) -> None:
    """Validate that an array contains only finite values.

    Args:
        array: The array to check
        name: Array name for error messages

    Raises:
        ValidationError: If array contains NaN or Inf
    """
    if not jnp.all(jnp.isfinite(array)):
        n_nan = jnp.sum(jnp.isnan(array))
        n_inf = jnp.sum(jnp.isinf(array))
        raise ValidationError(
            f"{name} contains non-finite values: {n_nan} NaN, {n_inf} Inf"
        )


def validate_density(n: Union[float, Array], name: str = "density") -> None:
    """Validate that density is physically reasonable.

    Args:
        n: Density value or array
        name: Parameter name for error messages

    Raises:
        ValidationError: If density is non-positive or unreasonably large
    """
    n_min = jnp.min(n) if hasattr(n, 'shape') else n
    n_max = jnp.max(n) if hasattr(n, 'shape') else n

    if n_min <= 0:
        raise ValidationError(f"{name} must be positive everywhere, min value: {n_min}")

    if n_max > 1e30:  # Reasonable upper bound for plasma density
        raise ValidationError(f"{name} exceeds physical bounds: max value {n_max}")


def validate_resistivity(eta: Union[float, Array], name: str = "resistivity") -> None:
    """Validate that resistivity is physically reasonable.

    Args:
        eta: Resistivity value or array
        name: Parameter name for error messages

    Raises:
        ValidationError: If resistivity is negative
    """
    eta_min = jnp.min(eta) if hasattr(eta, 'shape') else eta

    if eta_min < 0:
        raise ValidationError(f"{name} must be non-negative, min value: {eta_min}")


def validate_timestep(dt: float, dx: float, dy: float, name: str = "timestep") -> None:
    """Validate that timestep is reasonable relative to grid spacing.

    Args:
        dt: Timestep
        dx, dy: Grid spacings
        name: Parameter name for error messages

    Raises:
        ValidationError: If timestep is non-positive or suspiciously large
    """
    if dt <= 0:
        raise ValidationError(f"{name} must be positive, got {dt}")

    dx_min = min(dx, dy)
    # Very rough CFL check - actual CFL depends on wave speeds
    if dt > 100 * dx_min:
        raise ValidationError(
            f"{name} ({dt}) seems large relative to grid spacing ({dx_min}). "
            "Check CFL condition."
        )


def validate_grid_dimensions(nr: int, nz: int) -> None:
    """Validate grid dimensions are reasonable.

    Args:
        nr, nz: Number of grid points

    Raises:
        ValidationError: If dimensions are invalid
    """
    if nr < 4:
        raise ValidationError(f"nr must be at least 4 for finite differences, got {nr}")
    if nz < 4:
        raise ValidationError(f"nz must be at least 4 for finite differences, got {nz}")
    if nr > 10000 or nz > 10000:
        raise ValidationError(f"Grid dimensions ({nr}, {nz}) seem unreasonably large")


def validate_particle_count(n_particles: int) -> None:
    """Validate particle count is reasonable.

    Args:
        n_particles: Number of particles

    Raises:
        ValidationError: If count is invalid
    """
    if n_particles < 1:
        raise ValidationError(f"n_particles must be at least 1, got {n_particles}")
    if n_particles > 1e9:
        raise ValidationError(f"n_particles ({n_particles}) exceeds reasonable limit")
