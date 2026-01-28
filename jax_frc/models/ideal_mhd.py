"""Ideal MHD model using Constrained Transport scheme.

This model is optimized for advection-dominated flows (Rm >> 1) where
numerical diffusion must be minimized. It uses:
1. Constrained Transport (CT) for the induction equation
2. No divergence cleaning (CT preserves div(B) = 0)
3. 4th-order spatial discretization
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.solvers.constrained_transport import induction_rhs_ct
from jax_frc.operators import curl_3d
from jax_frc.constants import MU0


@dataclass(frozen=True)
class IdealMHD(PhysicsModel):
    """Ideal MHD model using Constrained Transport.

    Solves: dB/dt = curl(v × B)

    Uses CT scheme to preserve div(B) = 0 exactly without cleaning.
    Optimized for high magnetic Reynolds number (Rm >> 1) flows.

    Args:
        eta: Small resistivity for numerical stability [Ohm*m]
    """
    eta: float = 0.0  # Ideal MHD: no resistivity

    @partial(jax.jit, static_argnums=(0, 2))
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute dB/dt using Constrained Transport scheme.

        Args:
            state: Current state with B and v fields
            geometry: 3D geometry

        Returns:
            State with B field containing dB/dt
        """
        B = state.B
        v = state.v

        if v is None:
            # No velocity: dB/dt = 0 for ideal MHD
            return state.replace(B=jnp.zeros_like(B))

        # Use CT scheme for ideal induction: dB/dt = curl(v × B)
        dB_dt = induction_rhs_ct(v, B, geometry)

        # Add small resistive term if eta > 0 (for numerical stability)
        if self.eta > 0:
            J = curl_3d(B, geometry) / MU0
            # Resistive term: -curl(eta * J) = -eta * curl(curl(B)) / mu0
            # For simplicity, use diffusion form: eta * laplacian(B) / mu0
            # This is equivalent for uniform eta
            from jax_frc.operators import laplacian_3d
            dB_dt = dB_dt + self.eta / MU0 * laplacian_3d(B, geometry)

        return state.replace(B=dB_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """CFL condition for ideal MHD advection.

        dt < dx / |v_max|
        """
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)

        if state.v is None:
            return float('inf')

        v_max = jnp.max(jnp.sqrt(jnp.sum(state.v**2, axis=-1)))
        return 0.5 * dx_min / jnp.maximum(v_max, 1e-10)

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """No constraints needed - CT preserves div(B) = 0."""
        return state
