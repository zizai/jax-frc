"""3D Resistive MHD model with direct B-field evolution."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d
from jax_frc.constants import MU0


@dataclass(frozen=True)
class ResistiveMHD(PhysicsModel):
    """Resistive MHD model evolving B directly.

    Solves: dB/dt = -curl(E)
    Where:  E = -v x B + eta*J, J = curl(B)/mu_0

    For stationary plasma (v=0): E = eta*J

    Args:
        eta: Resistivity [Ohm*m]
    """
    eta: float = 1e-4  # Resistivity [Ohm*m]

    @partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute dB/dt from induction equation.

        Args:
            state: Current state with B field
            geometry: 3D geometry

        Returns:
            State with B field containing dB/dt
        """
        B = state.B

        # Current density: J = curl(B) / mu_0
        J = curl_3d(B, geometry) / MU0

        # Electric field: E = eta*J (assuming v=0 for pure resistive case)
        # For moving plasma: E = -v x B + eta*J
        if state.v is not None:
            v = state.v
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            E = -v_cross_B + self.eta * J
        else:
            E = self.eta * J

        # Faraday's law: dB/dt = -curl(E)
        dB_dt = -curl_3d(E, geometry)

        return state.replace(B=dB_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Resistive diffusion CFL: dt < dx^2 / (2*eta/mu0)."""
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)
        diffusivity = self.eta / MU0
        return 0.25 * dx_min**2 / diffusivity

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions based on geometry.bc_* settings."""
        # For now, return state unchanged (periodic BCs handled by operators)
        return state
