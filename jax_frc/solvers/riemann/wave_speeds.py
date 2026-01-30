"""MHD wave speed calculations.

Computes characteristic wave speeds for MHD systems:
- Fast magnetosonic speed
- Alfven speed
- Slow magnetosonic speed

These are used for CFL conditions and Riemann solver wave estimates.
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple


@jit
def mhd_wave_speeds(
    rho: jnp.ndarray,
    p: jnp.ndarray,
    Bx: jnp.ndarray,
    By: jnp.ndarray,
    Bz: jnp.ndarray,
    gamma: float = 5.0 / 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute MHD characteristic wave speeds.

    Args:
        rho: Density
        p: Pressure
        Bx: Magnetic field x-component (normal direction)
        By: Magnetic field y-component (tangential)
        Bz: Magnetic field z-component (tangential)
        gamma: Adiabatic index (default 5/3)

    Returns:
        Tuple of (c_fast, c_alfven, c_slow) wave speeds
    """
    # Ensure positive values for stability
    rho_safe = jnp.maximum(rho, 1e-12)
    p_safe = jnp.maximum(p, 1e-12)

    # Sound speed squared
    a2 = gamma * p_safe / rho_safe

    # Alfven speed squared (total and normal)
    B2 = Bx**2 + By**2 + Bz**2
    va2 = B2 / rho_safe  # Total Alfven speed squared
    vax2 = Bx**2 / rho_safe  # Normal Alfven speed squared

    # Fast and slow magnetosonic speeds
    # c_f,s^2 = 0.5 * (a^2 + va^2 +/- sqrt((a^2 + va^2)^2 - 4*a^2*vax^2))
    sum_sq = a2 + va2
    discriminant = jnp.maximum(sum_sq**2 - 4.0 * a2 * vax2, 0.0)
    sqrt_disc = jnp.sqrt(discriminant)

    cf2 = 0.5 * (sum_sq + sqrt_disc)
    cs2 = 0.5 * (sum_sq - sqrt_disc)

    c_fast = jnp.sqrt(jnp.maximum(cf2, 0.0))
    c_alfven = jnp.sqrt(jnp.maximum(vax2, 0.0))
    c_slow = jnp.sqrt(jnp.maximum(cs2, 0.0))

    return c_fast, c_alfven, c_slow


@jit
def fast_magnetosonic_speed(
    rho: jnp.ndarray,
    p: jnp.ndarray,
    Bx: jnp.ndarray,
    By: jnp.ndarray,
    Bz: jnp.ndarray,
    gamma: float = 5.0 / 3.0,
) -> jnp.ndarray:
    """Compute fast magnetosonic speed for CFL condition.

    Args:
        rho: Density
        p: Pressure
        Bx, By, Bz: Magnetic field components
        gamma: Adiabatic index

    Returns:
        Fast magnetosonic speed
    """
    c_fast, _, _ = mhd_wave_speeds(rho, p, Bx, By, Bz, gamma)
    return c_fast
