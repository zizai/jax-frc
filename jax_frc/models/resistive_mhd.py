"""3D Resistive MHD model with full field evolution.

Solves the full ideal/resistive MHD equations:
    - Continuity: dn/dt = -div(n*v)
    - Momentum:   dv/dt = -grad(p)/rho + J×B/rho - (v·grad)v
    - Pressure:   dp/dt = -gamma*p*div(v)
    - Induction:  dB/dt = curl(v×B) + eta*laplacian(B)/mu0
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Literal
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d, laplacian_3d, divergence_3d, gradient_3d
from jax_frc.constants import MU0, MI
from jax_frc.fields import CoilField


AdvectionScheme = Literal["central", "ct", "skew_symmetric", "hll"]

# Adiabatic index for ideal gas
GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class ResistiveMHD(PhysicsModel):
    """Full resistive MHD model evolving all fields.

    Solves the complete MHD system:
        - Continuity: dn/dt = -div(n*v)
        - Momentum:   dv/dt = -grad(p)/rho + J×B/rho - (v·grad)v
        - Pressure:   dp/dt = -gamma*p*div(v)
        - Induction:  dB/dt = curl(v×B) + eta*laplacian(B)/mu0

    Where rho = m_i * n (mass density), J = curl(B)/mu0 (current density).

    Args:
        eta: Resistivity [Ohm*m] (or dimensionless if normalized_units=True)
        advection_scheme: Numerical scheme for advection term curl(v×B).
            - "central": Standard central differences (default, backward compatible)
            - "ct": Constrained Transport scheme (preserves div(B)=0, less diffusive)
            - "skew_symmetric": Energy-conserving skew-symmetric formulation
            - "hll": HLL Riemann solver with PLM reconstruction (AGATE-compatible)
        evolve_density: Whether to evolve density (default True)
        evolve_velocity: Whether to evolve velocity (default True)
        evolve_pressure: Whether to evolve pressure (default True)
        normalized_units: If True, use normalized/dimensionless units where n is
            mass density directly (rho=n). If False (default), use SI units where
            n is particle number density (rho=m_i*n). Set True for standard MHD
            test problems like Orszag-Tang.
        hll_beta: MC limiter parameter for HLL reconstruction (default 1.3, AGATE default)
    """
    eta: float = 1e-4  # Resistivity [Ohm*m]
    external_field: Optional[CoilField] = None
    advection_scheme: AdvectionScheme = "central"
    evolve_density: bool = True
    evolve_velocity: bool = True
    evolve_pressure: bool = True
    normalized_units: bool = False
    hll_beta: float = 1.3  # MC limiter parameter for HLL (AGATE default)

    @partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all MHD fields.

        Args:
            state: Current state with n, v, p, B fields
            geometry: 3D geometry

        Returns:
            State with fields containing their time derivatives
        """
        n = state.n
        p = state.p
        B = state.B
        v = state.v if state.v is not None else jnp.zeros((*n.shape, 3))

        # Mass density: rho = n for normalized units, rho = m_i * n for SI units
        if self.normalized_units:
            rho = n
        else:
            rho = MI * n
        # Avoid division by zero
        rho_safe = jnp.maximum(rho, 1e-20)

        # =====================================================================
        # Continuity equation: dn/dt = -div(n*v)
        # =====================================================================
        if self.evolve_density:
            # Compute n*v flux
            nv = n[..., None] * v  # (nx, ny, nz, 3)
            div_nv = divergence_3d(nv, geometry)
            dn_dt = -div_nv
        else:
            dn_dt = jnp.zeros_like(n)

        # =====================================================================
        # Momentum equation: dv/dt = -grad(p)/rho + J×B/rho - (v·grad)v
        # =====================================================================
        if self.evolve_velocity:
            # Pressure gradient force: -grad(p)/rho
            grad_p = gradient_3d(p, geometry)  # (nx, ny, nz, 3)
            pressure_force = -grad_p / rho_safe[..., None]

            # Lorentz force: J×B/rho where J = curl(B)/mu0 (SI) or J = curl(B) (normalized)
            if self.normalized_units:
                J = curl_3d(B, geometry)  # Normalized: mu0 = 1
            else:
                J = curl_3d(B, geometry) / MU0  # SI units
            # J×B cross product
            JxB = jnp.stack([
                J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
                J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
                J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
            ], axis=-1)
            lorentz_force = JxB / rho_safe[..., None]

            # Advection term: -(v·grad)v
            # Compute each component: (v·grad)v_i = v_x * dv_i/dx + v_y * dv_i/dy + v_z * dv_i/dz
            grad_vx = gradient_3d(v[..., 0], geometry)
            grad_vy = gradient_3d(v[..., 1], geometry)
            grad_vz = gradient_3d(v[..., 2], geometry)

            v_dot_grad_v = jnp.stack([
                v[..., 0] * grad_vx[..., 0] + v[..., 1] * grad_vx[..., 1] + v[..., 2] * grad_vx[..., 2],
                v[..., 0] * grad_vy[..., 0] + v[..., 1] * grad_vy[..., 1] + v[..., 2] * grad_vy[..., 2],
                v[..., 0] * grad_vz[..., 0] + v[..., 1] * grad_vz[..., 1] + v[..., 2] * grad_vz[..., 2],
            ], axis=-1)

            dv_dt = pressure_force + lorentz_force - v_dot_grad_v
        else:
            dv_dt = jnp.zeros_like(v)

        # =====================================================================
        # Pressure equation: dp/dt = -gamma*p*div(v)
        # =====================================================================
        if self.evolve_pressure:
            div_v = divergence_3d(v, geometry)
            dp_dt = -GAMMA * p * div_v
        else:
            dp_dt = jnp.zeros_like(p)

        # =====================================================================
        # Induction equation: dB/dt = curl(v×B) + eta*laplacian(B)/mu0
        # =====================================================================
        # Resistive term: eta * laplacian(B) / mu0 (SI) or eta * laplacian(B) (normalized)
        # Apply Laplacian component-by-component (vector Laplacian)
        if self.normalized_units:
            diffusion_coeff = self.eta  # Normalized: mu0 = 1
        else:
            diffusion_coeff = self.eta / MU0  # SI units
        dB_dt_resistive = diffusion_coeff * jnp.stack([
            laplacian_3d(B[..., 0], geometry),
            laplacian_3d(B[..., 1], geometry),
            laplacian_3d(B[..., 2], geometry),
        ], axis=-1)

        # Advection term: curl(v × B)
        if self.advection_scheme == "ct":
            # Constrained Transport scheme - preserves div(B)=0
            from jax_frc.solvers.constrained_transport import induction_rhs_ct
            dB_dt_advection = induction_rhs_ct(v, B, geometry)
        elif self.advection_scheme == "skew_symmetric":
            # Energy-conserving skew-symmetric formulation
            from jax_frc.solvers.constrained_transport import induction_rhs_skew_symmetric
            dB_dt_advection = induction_rhs_skew_symmetric(v, B, geometry)
        elif self.advection_scheme == "hll":
            # HLL Riemann solver with PLM reconstruction (AGATE-compatible)
            from jax_frc.solvers.riemann import hll_flux_3d
            # Sum contributions from all directions
            dB_dt_x = hll_flux_3d(state, geometry, GAMMA, self.hll_beta, direction=0)
            dB_dt_z = hll_flux_3d(state, geometry, GAMMA, self.hll_beta, direction=2)
            # For 2D (ny=1), skip y-direction
            if geometry.ny > 1:
                dB_dt_y = hll_flux_3d(state, geometry, GAMMA, self.hll_beta, direction=1)
                dB_dt_advection = dB_dt_x + dB_dt_y + dB_dt_z
            else:
                dB_dt_advection = dB_dt_x + dB_dt_z
        else:
            # Standard central difference scheme
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            dB_dt_advection = curl_3d(v_cross_B, geometry)

        dB_dt = dB_dt_advection + dB_dt_resistive

        return state.replace(n=dn_dt, v=dv_dt, p=dp_dt, B=dB_dt)

    def explicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Return explicit IMEX terms (hyperbolic terms for all fields).

        For IMEX splitting, this returns the explicit (hyperbolic) part:
        - Continuity: -div(n*v)
        - Momentum: -grad(p)/rho + J×B/rho - (v·grad)v
        - Pressure: -gamma*p*div(v)
        - Induction: curl(v×B)
        """
        n = state.n
        p = state.p
        B = state.B
        v = state.v if state.v is not None else jnp.zeros((*n.shape, 3))

        zero_E = jnp.zeros_like(state.E) if state.E is not None else None
        zero_Te = jnp.zeros_like(state.Te) if state.Te is not None else None

        # Mass density: rho = n for normalized units, rho = m_i * n for SI units
        if self.normalized_units:
            rho = n
        else:
            rho = MI * n
        rho_safe = jnp.maximum(rho, 1e-20)

        # Continuity: dn/dt = -div(n*v)
        if self.evolve_density:
            nv = n[..., None] * v
            dn_dt = -divergence_3d(nv, geometry)
        else:
            dn_dt = jnp.zeros_like(n)

        # Momentum: dv/dt = -grad(p)/rho + J×B/rho - (v·grad)v
        if self.evolve_velocity:
            grad_p = gradient_3d(p, geometry)
            pressure_force = -grad_p / rho_safe[..., None]

            if self.normalized_units:
                J = curl_3d(B, geometry)  # Normalized: mu0 = 1
            else:
                J = curl_3d(B, geometry) / MU0  # SI units
            JxB = jnp.stack([
                J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
                J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
                J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
            ], axis=-1)
            lorentz_force = JxB / rho_safe[..., None]

            grad_vx = gradient_3d(v[..., 0], geometry)
            grad_vy = gradient_3d(v[..., 1], geometry)
            grad_vz = gradient_3d(v[..., 2], geometry)
            v_dot_grad_v = jnp.stack([
                v[..., 0] * grad_vx[..., 0] + v[..., 1] * grad_vx[..., 1] + v[..., 2] * grad_vx[..., 2],
                v[..., 0] * grad_vy[..., 0] + v[..., 1] * grad_vy[..., 1] + v[..., 2] * grad_vy[..., 2],
                v[..., 0] * grad_vz[..., 0] + v[..., 1] * grad_vz[..., 1] + v[..., 2] * grad_vz[..., 2],
            ], axis=-1)

            dv_dt = pressure_force + lorentz_force - v_dot_grad_v
        else:
            dv_dt = jnp.zeros_like(v)

        # Pressure: dp/dt = -gamma*p*div(v)
        if self.evolve_pressure:
            div_v = divergence_3d(v, geometry)
            dp_dt = -GAMMA * p * div_v
        else:
            dp_dt = jnp.zeros_like(p)

        # Induction: dB/dt = curl(v×B) (explicit part only)
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

        return state.replace(n=dn_dt, v=dv_dt, p=dp_dt, B=dB_dt, E=zero_E, Te=zero_Te)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Compute stable timestep based on CFL conditions.

        Considers:
        - Alfven wave CFL: dt < dx / v_A where v_A = B / sqrt(mu0 * rho)
        - Sound wave CFL: dt < dx / c_s where c_s = sqrt(gamma * p / rho)
        - Resistive diffusion CFL: dt < dx^2 / (2*eta/mu0)
        """
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)

        # Resistive diffusion CFL
        if self.normalized_units:
            diffusivity = self.eta  # Normalized: mu0 = 1
        else:
            diffusivity = self.eta / MU0  # SI units
        dt_resistive = 0.25 * dx_min**2 / max(diffusivity, 1e-20)

        # Alfven wave CFL
        B_mag = jnp.sqrt(jnp.sum(state.B**2, axis=-1))
        if self.normalized_units:
            rho = state.n
        else:
            rho = MI * state.n
        rho_safe = jnp.maximum(rho, 1e-20)
        if self.normalized_units:
            v_alfven = jnp.max(B_mag / jnp.sqrt(rho_safe))  # Normalized: mu0 = 1
        else:
            v_alfven = jnp.max(B_mag / jnp.sqrt(MU0 * rho_safe))  # SI units
        dt_alfven = dx_min / max(float(v_alfven), 1e-10)

        # Sound wave CFL
        c_sound = jnp.max(jnp.sqrt(GAMMA * state.p / rho_safe))
        dt_sound = dx_min / max(float(c_sound), 1e-10)

        # Flow CFL
        if state.v is not None:
            v_mag = jnp.max(jnp.sqrt(jnp.sum(state.v**2, axis=-1)))
            dt_flow = dx_min / max(float(v_mag), 1e-10)
        else:
            dt_flow = float('inf')

        # Return minimum with safety factor
        return 0.5 * min(dt_resistive, dt_alfven, dt_sound, dt_flow)

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
