"""Neutral fluid model for plasma-neutral coupling.

Implements Euler equations for neutral gas with atomic source terms.
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit

from jax_frc.constants import MI

# Adiabatic index for monatomic gas
GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class NeutralState:
    """Neutral fluid state variables.

    All fields use SI units.
    """

    rho_n: Array  # Mass density [kg/m³], shape (nr, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nr, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nr, nz)

    @property
    def v_n(self) -> Array:
        """Velocity [m/s], shape (nr, nz, 3)."""
        rho_safe = jnp.maximum(self.rho_n[..., None], 1e-20)
        return self.mom_n / rho_safe

    @property
    def p_n(self) -> Array:
        """Pressure [Pa] from ideal gas EOS, shape (nr, nz)."""
        rho_safe = jnp.maximum(self.rho_n, 1e-20)
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / rho_safe
        internal_energy = self.E_n - ke
        return (GAMMA - 1) * jnp.maximum(internal_energy, 0.0)

    @property
    def T_n(self) -> Array:
        """Temperature [J], shape (nr, nz)."""
        n_n = self.rho_n / MI
        n_safe = jnp.maximum(n_n, 1e-10)
        return self.p_n / n_safe

    def replace(self, **kwargs) -> "NeutralState":
        """Return new NeutralState with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register NeutralState as JAX pytree
def _neutral_state_flatten(state):
    children = (state.rho_n, state.mom_n, state.E_n)
    aux_data = None
    return children, aux_data


def _neutral_state_unflatten(aux_data, children):
    rho_n, mom_n, E_n = children
    return NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)


jax.tree_util.register_pytree_node(
    NeutralState, _neutral_state_flatten, _neutral_state_unflatten
)


@jit
def euler_flux_1d(rho: Array, v: Array, p: Array, E: Array) -> Tuple[Array, Array, Array]:
    """Compute 1D Euler fluxes.

    Args:
        rho: Mass density [kg/m³]
        v: Velocity component in flux direction [m/s]
        p: Pressure [Pa]
        E: Total energy density [J/m³]

    Returns:
        F_rho: Mass flux [kg/m²/s]
        F_mom: Momentum flux [Pa]
        F_E: Energy flux [W/m²]
    """
    F_rho = rho * v
    F_mom = rho * v**2 + p
    F_E = (E + p) * v
    return F_rho, F_mom, F_E
