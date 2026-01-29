"""3D Resistive MHD model with direct B-field evolution."""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Literal
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d, laplacian_3d
from jax_frc.constants import MU0
from jax_frc.fields import CoilField


AdvectionScheme = Literal["central", "ct", "skew_symmetric"]


@dataclass(frozen=True)
class ResistiveMHD(PhysicsModel):
    """Resistive MHD model evolving B directly.

    Solves: dB/dt = -curl(E)
    Where:  E = -v x B + eta*J, J = curl(B)/mu_0

    For stationary plasma (v=0): E = eta*J

    Args:
        eta: Resistivity [Ohm*m]
        advection_scheme: Numerical scheme for advection term curl(v×B).
            - "central": Standard central differences (default, backward compatible)
            - "ct": Constrained Transport scheme (preserves div(B)=0, less diffusive)
            - "skew_symmetric": Energy-conserving skew-symmetric formulation
    """
    eta: float = 1e-4  # Resistivity [Ohm*m]
    external_field: Optional[CoilField] = None
    advection_scheme: AdvectionScheme = "central"

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

        # Resistive term: eta * laplacian(B) / mu0
        # (equivalent to -curl(eta*J) for uniform eta)
        dB_dt_resistive = self.eta / MU0 * laplacian_3d(B, geometry)

        # Advection term: curl(v × B)
        if state.v is not None:
            v = state.v
            if self.advection_scheme == "ct":
                # Constrained Transport scheme - preserves div(B)=0
                from jax_frc.solvers.constrained_transport import induction_rhs_ct
                dB_dt_advection = induction_rhs_ct(v, B, geometry)
            elif self.advection_scheme == "skew_symmetric":
                # Energy-conserving skew-symmetric formulation
                from jax_frc.solvers.constrained_transport import induction_rhs_skew_symmetric
                dB_dt_advection = induction_rhs_skew_symmetric(v, B, geometry)
            else:
                # Standard central difference scheme
                v_cross_B = jnp.stack([
                    v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                    v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                    v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
                ], axis=-1)
                dB_dt_advection = curl_3d(v_cross_B, geometry)
        else:
            dB_dt_advection = jnp.zeros_like(B)

        dB_dt = dB_dt_advection + dB_dt_resistive

        return state.replace(B=dB_dt)

    def explicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Return explicit IMEX terms (ideal induction via v x B)."""
        zero_E = jnp.zeros_like(state.E) if state.E is not None else None
        zero_Te = jnp.zeros_like(state.Te) if state.Te is not None else None

        if state.v is None:
            zero_B = jnp.zeros_like(state.B)
            return state.replace(B=zero_B, E=zero_E, Te=zero_Te)

        B = state.B
        v = state.v

        if self.advection_scheme == "ct":
            from jax_frc.solvers.constrained_transport import induction_rhs_ct
            dB_dt = induction_rhs_ct(v, B, geometry)
        elif self.advection_scheme == "skew_symmetric":
            from jax_frc.solvers.constrained_transport import induction_rhs_skew_symmetric
            dB_dt = induction_rhs_skew_symmetric(v, B, geometry)
        else:
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            dB_dt = curl_3d(v_cross_B, geometry)

        return state.replace(B=dB_dt, E=zero_E, Te=zero_Te)

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

        Note: When using CT scheme, div(B)=0 is preserved exactly, so cleaning
        is skipped to avoid introducing numerical errors.
        """
        # CT and skew_symmetric schemes preserve div(B)=0 exactly - no cleaning needed
        if self.advection_scheme in ("ct", "skew_symmetric"):
            return state

        from jax_frc.solvers.divergence_cleaning import clean_divergence

        B_clean = clean_divergence(state.B, geometry, max_iter=200, tol=1e-6)
        return state.replace(B=B_clean)
