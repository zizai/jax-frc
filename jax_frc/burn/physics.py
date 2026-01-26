"""Fusion reaction rates and power calculations.

Implements Bosch-Hale parameterization for D-T, D-D, and D-3He reactions.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array, jit

from jax_frc.constants import (
    BOSCH_HALE_DT, BOSCH_HALE_DD_T, BOSCH_HALE_DD_HE3, BOSCH_HALE_DHE3,
)

ReactionType = Literal["DT", "DD_T", "DD_HE3", "DHE3"]


@jit(static_argnums=(1,))
def reactivity(T_keV: Array, reaction: str) -> Array:
    """Compute fusion reactivity <sigma*v> using Bosch-Hale parameterization.

    Args:
        T_keV: Temperature in keV (scalar or array)
        reaction: Reaction type ("DT", "DD_T", "DD_HE3", "DHE3")

    Returns:
        Reactivity in cm^3/s
    """
    # Get coefficients based on reaction type
    if reaction == "DT":
        coef = BOSCH_HALE_DT
    elif reaction == "DD_T":
        coef = BOSCH_HALE_DD_T
    elif reaction == "DD_HE3":
        coef = BOSCH_HALE_DD_HE3
    elif reaction == "DHE3":
        coef = BOSCH_HALE_DHE3
    else:
        raise ValueError(f"Unknown reaction: {reaction}")

    B_G = coef["B_G"]
    m_rc2 = coef["m_rc2"]
    C1, C2, C3 = coef["C1"], coef["C2"], coef["C3"]
    C4, C5, C6, C7 = coef["C4"], coef["C5"], coef["C6"], coef["C7"]

    # Ensure T is positive to avoid numerical issues
    T = jnp.maximum(T_keV, 0.1)

    # Compute theta
    numerator = T * (C2 + T * (C4 + T * C6))
    denominator = 1.0 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1.0 - numerator / denominator)

    # Compute xi
    xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)

    # Compute <sigma*v>
    sigma_v = C1 * theta * jnp.sqrt(xi / (m_rc2 * T**3)) * jnp.exp(-3.0 * xi)

    return sigma_v


@dataclass
class BurnPhysics:
    """Placeholder for burn physics calculations.

    This class will be implemented in Task 3.
    """
    pass
