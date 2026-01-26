"""Atomic rate coefficients for plasma-neutral interactions.

Includes ionization, recombination, charge exchange, and radiation.
All rates use SI units and are JIT-compatible.
"""

import jax.numpy as jnp
from jax import jit, Array

from jax_frc.constants import QE


# =============================================================================
# Ionization (electron impact: H + e -> H+ + 2e)
# =============================================================================

@jit
def ionization_rate_coefficient(Te: Array) -> Array:
    """Voronov fit for hydrogen ionization <sigma*v>_ion(Te) [m^3/s].

    Reference: Voronov (1997), Atomic Data and Nuclear Data Tables 65, 1-35.

    <sigma*v> = A * (1 + P*sqrt(U)) * U^K * exp(-U) / (X + U)
    where U = E_ion / Te, E_ion = 13.6 eV

    Args:
        Te: Electron temperature [J] (can be array)

    Returns:
        Rate coefficient [m^3/s]
    """
    E_ion = 13.6 * QE  # Ionization energy in Joules

    # Clamp Te to avoid division by zero and overflow
    Te_safe = jnp.maximum(Te, 0.1 * QE)  # Min 0.1 eV

    U = E_ion / Te_safe

    # Voronov coefficients for hydrogen
    A = 2.91e-14  # m^3/s
    P = 0.0
    K = 0.39
    X = 0.232

    # Clamp U to prevent overflow in exp(-U)
    U_clamped = jnp.minimum(U, 100.0)

    sigma_v = A * (1 + P * jnp.sqrt(U_clamped)) * U_clamped**K * jnp.exp(-U_clamped) / (X + U_clamped)

    return sigma_v
