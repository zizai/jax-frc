"""3D equilibrium initializers.

Provides initial magnetic field configurations for 3D simulations.
All functions return vector fields with shape (nx, ny, nz, 3).
"""

import jax.numpy as jnp
from jax import Array

from jax_frc.core.geometry import Geometry


def harris_sheet_3d(geometry: Geometry, B0: float = 0.1, L: float = 0.1) -> Array:
    """Harris current sheet: Bx = B0 * tanh(y / L).

    A simple 1D magnetic configuration with current flowing in the z direction.
    The field reverses sign across y=0.

    Args:
        geometry: 3D Cartesian geometry
        B0: Asymptotic magnetic field strength [T]
        L: Current sheet thickness [m]

    Returns:
        Magnetic field array, shape (nx, ny, nz, 3)
    """
    y = geometry.y_grid
    Bx = B0 * jnp.tanh(y / L)
    By = jnp.zeros_like(y)
    Bz = jnp.zeros_like(y)
    return jnp.stack([Bx, By, Bz], axis=-1)


def uniform_field_3d(geometry: Geometry, B0: float = 0.1, direction: str = "z") -> Array:
    """Uniform magnetic field in a specified direction.

    Args:
        geometry: 3D Cartesian geometry
        B0: Magnetic field strength [T]
        direction: Field direction, one of "x", "y", or "z"

    Returns:
        Magnetic field array, shape (nx, ny, nz, 3)
    """
    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    idx = {"x": 0, "y": 1, "z": 2}[direction]
    return B.at[..., idx].set(B0)


def flux_rope_3d(geometry: Geometry, B0: float = 0.1, a: float = 0.3) -> Array:
    """Cylindrical flux rope (FRC-like) centered on z-axis.

    Creates an axial field Bz with a parabolic profile that vanishes at r=a,
    and an azimuthal component B_theta that provides twist.

    Args:
        geometry: 3D Cartesian geometry
        B0: Peak axial magnetic field strength [T]
        a: Core radius [m] - field vanishes for r > a

    Returns:
        Magnetic field array, shape (nx, ny, nz, 3)
    """
    x, y = geometry.x_grid, geometry.y_grid
    r = jnp.sqrt(x**2 + y**2)

    # Axial field: parabolic profile, vanishes at r=a
    Bz = B0 * jnp.maximum(1 - (r / a)**2, 0)

    # Azimuthal field component for twist
    theta = jnp.arctan2(y, x)
    B_theta = 0.1 * B0 * (r / a) * jnp.where(r < a, 1.0, 0.0)

    # Convert to Cartesian components
    Bx = -B_theta * jnp.sin(theta)
    By = B_theta * jnp.cos(theta)

    return jnp.stack([Bx, By, Bz], axis=-1)
