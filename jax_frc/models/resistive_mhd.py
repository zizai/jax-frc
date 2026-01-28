"""3D Resistive MHD model with direct B-field evolution."""

from dataclasses import dataclass
from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d
from jax_frc.constants import MU0
from jax_frc.fields import CoilField


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
    external_field: Optional[CoilField] = None

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

    def explicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Return explicit terms for IMEX (none for pure resistive diffusion)."""
        zero_B = jnp.zeros_like(state.B)
        zero_E = jnp.zeros_like(state.E) if state.E is not None else None
        zero_Te = jnp.zeros_like(state.Te) if state.Te is not None else None
        return state.replace(B=zero_B, E=zero_E, Te=zero_Te)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Resistive diffusion CFL: dt < dx^2 / (2*eta/mu0)."""
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)
        diffusivity = self.eta / MU0
        return 0.25 * dx_min**2 / diffusivity

    def get_total_B(
        self, state: State, geometry: Geometry, t: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return total B components (Bx, Bz), including external field."""
        B_total = state.B
        if self.external_field is not None:
            r = jnp.abs(geometry.x_grid)
            z = geometry.z_grid
            B_r, B_z = self.external_field.B_field(r, z, t)
            B_ext = jnp.zeros_like(B_total)
            B_ext = B_ext.at[:, :, :, 0].set(B_r)
            B_ext = B_ext.at[:, :, :, 2].set(B_z)
            B_total = B_total + B_ext

        return B_total[:, :, :, 0], B_total[:, :, :, 2]

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply divergence cleaning to magnetic field.

        Uses projection method: B_clean = B - grad(phi) where laplacian(phi) = div(B)
        """
        if not (
            geometry.bc_x == "periodic"
            and geometry.bc_y == "periodic"
            and geometry.bc_z == "periodic"
        ):
            return state

        from jax_frc.solvers.divergence_cleaning import clean_divergence

        B_clean = clean_divergence(state.B, geometry, max_iter=200, tol=1e-6)
        return state.replace(B=B_clean)
