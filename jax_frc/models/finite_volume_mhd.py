"""Finite Volume MHD model with coupled evolution.

This model implements a proper finite volume method for MHD,
evolving all conserved variables together using Riemann solvers.

Unlike the standard ResistiveMHD model which evolves fields separately,
this model uses a Godunov-type finite volume approach for stability
and accuracy.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.solvers.riemann.mhd_state import (
    MHDConserved,
    MHDPrimitive,
    primitive_to_conserved,
    conserved_to_primitive,
    state_to_conserved,
    conserved_to_state,
)
from jax_frc.solvers.riemann.hll_full import hll_update_full
from jax_frc.solvers.riemann.hlld import hlld_update_full
from jax_frc.solvers.riemann.dedner import glm_cleaning_update
from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed


RiemannSolver = Literal["hll", "hlld"]
Reconstruction = Literal["plm", "ppm"]


@dataclass(frozen=True)
class FiniteVolumeMHD(PhysicsModel):
    """Finite volume MHD model with coupled evolution.

    This model solves the ideal MHD equations using a Godunov-type
    finite volume method with approximate Riemann solvers.

    Conservation laws:
        d(rho)/dt + div(rho*v) = 0
        d(rho*v)/dt + div(rho*v*v + p_tot*I - B*B) = 0
        d(E)/dt + div((E + p_tot)*v - B*(v.B)) = 0
        dB/dt + curl(E) = 0, where E = -v x B

    Args:
        gamma: Adiabatic index (default 5/3)
        riemann_solver: "hll" or "hlld" (default)
            - "hll": HLL 2-wave solver (robust, diffusive)
            - "hlld": HLLD 5-wave solver (accurate)
        reconstruction: "plm" (default) or "ppm"
        limiter_beta: MC limiter parameter (default 1.3, AGATE default)
        cfl: CFL number for time step (default 0.4)
    """
    gamma: float = 5.0 / 3.0
    riemann_solver: RiemannSolver = "hll"
    reconstruction: Reconstruction = "plm"
    limiter_beta: float = 1.3
    cfl: float = 0.4
    evolve_density: bool = True
    evolve_velocity: bool = True
    evolve_pressure: bool = True
    use_divergence_cleaning: bool = False
    cleaning_speed: float | None = None
    cleaning_cr: float = 0.18

    @partial(jax.jit, static_argnums=(0, 2))
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives using finite volume method.

        Args:
            state: Current state with n, v, p, B fields
            geometry: 3D geometry

        Returns:
            State with fields containing their time derivatives
        """
        # Convert to conserved variables
        cons = state_to_conserved(state, self.gamma)

        # Compute update using Riemann solver
        if self.riemann_solver == "hll":
            dU = hll_update_full(cons, geometry, self.gamma, self.limiter_beta)
        elif self.riemann_solver == "hlld":
            dU = hlld_update_full(cons, geometry, self.gamma, self.limiter_beta)
        else:
            # Default to HLL
            dU = hll_update_full(cons, geometry, self.gamma, self.limiter_beta)

        # Convert dU back to State format
        # dU contains d(rho)/dt, d(mom)/dt, d(E)/dt, d(B)/dt
        # We need to convert to dn/dt, dv/dt, dp/dt, dB/dt

        # For density: dn/dt = d(rho)/dt (since n = rho in normalized units)
        dn_dt = dU.rho

        # For velocity: dv/dt = (d(mom)/dt - v * d(rho)/dt) / rho
        rho = jnp.maximum(state.n, 1e-12)
        v = state.v if state.v is not None else jnp.zeros((*rho.shape, 3))
        dv_dt = jnp.stack([
            (dU.mom_x - v[..., 0] * dU.rho) / rho,
            (dU.mom_y - v[..., 1] * dU.rho) / rho,
            (dU.mom_z - v[..., 2] * dU.rho) / rho,
        ], axis=-1)

        # For pressure: dp/dt from energy equation
        # E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
        # dE/dt = dp/dt/(gamma-1) + 0.5*d(rho*v^2)/dt + 0.5*d(B^2)/dt
        # dp/dt = (gamma-1) * (dE/dt - 0.5*d(rho*v^2)/dt - 0.5*d(B^2)/dt)
        v2 = jnp.sum(v**2, axis=-1)
        B = state.B
        B2 = jnp.sum(B**2, axis=-1)

        # d(rho*v^2)/dt = v^2 * d(rho)/dt + 2*rho*v.dv/dt
        d_rho_v2_dt = v2 * dU.rho + 2 * rho * jnp.sum(v * dv_dt, axis=-1)

        # d(B^2)/dt = 2*B.dB/dt
        dB_dt = jnp.stack([dU.Bx, dU.By, dU.Bz], axis=-1)
        d_B2_dt = 2 * jnp.sum(B * dB_dt, axis=-1)

        dp_dt = (self.gamma - 1.0) * (dU.E - 0.5 * d_rho_v2_dt - 0.5 * d_B2_dt)

        if not self.evolve_density:
            dn_dt = jnp.zeros_like(state.n)
        if not self.evolve_velocity:
            dv_dt = jnp.zeros_like(v)
        if not self.evolve_pressure:
            dp_dt = jnp.zeros_like(state.p)

        dpsi_dt = None
        if self.use_divergence_cleaning:
            psi = state.psi if state.psi is not None else jnp.zeros_like(state.n)
            if self.cleaning_speed is None:
                rho = jnp.maximum(state.n, 1e-12)
                p = jnp.maximum(state.p, 1e-12)
                v_mag = jnp.sqrt(jnp.sum(v**2, axis=-1))
                cf = fast_magnetosonic_speed(
                    rho, p, B[..., 0], B[..., 1], B[..., 2], self.gamma
                )
                ch = 0.95 * jnp.max(cf + v_mag)
            else:
                ch = jnp.asarray(self.cleaning_speed)
            dB_clean, dpsi_dt = glm_cleaning_update(
                B, geometry, psi, ch, self.limiter_beta
            )
            dB_dt = dB_dt + dB_clean
            dpsi_dt = dpsi_dt - (ch / self.cleaning_cr) * psi

        if dpsi_dt is None:
            return state.replace(n=dn_dt, v=dv_dt, p=dp_dt, B=dB_dt)
        return state.replace(n=dn_dt, v=dv_dt, p=dp_dt, B=dB_dt, psi=dpsi_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Compute stable time step based on CFL condition.

        Args:
            state: Current state
            geometry: Grid geometry

        Returns:
            Stable time step
        """
        rho = jnp.maximum(state.n, 1e-12)
        p = jnp.maximum(state.p, 1e-12)
        B = state.B
        v = state.v

        # Fast magnetosonic speed
        Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
        B_mag = jnp.sqrt(Bx**2 + By**2 + Bz**2)
        cf = fast_magnetosonic_speed(rho, p, Bx, By, Bz, self.gamma)

        # Maximum wave speed including flow velocity
        v_mag = jnp.sqrt(jnp.sum(v**2, axis=-1))
        max_speed = jnp.max(cf + v_mag)

        # Minimum grid spacing
        dx_min = min(geometry.dx, geometry.dz)
        if geometry.ny > 1:
            dx_min = min(dx_min, geometry.dy)

        # CFL condition
        dt = self.cfl * dx_min / max(float(max_speed), 1e-10)

        return dt

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply physical constraints.

        For finite volume MHD, we apply:
        - Density and pressure floors

        Args:
            state: Current state
            geometry: Grid geometry

        Returns:
            State with constraints applied
        """
        # Apply density and pressure floors
        n = jnp.maximum(state.n, 1e-12)
        p = jnp.maximum(state.p, 1e-12)

        return state.replace(n=n, p=p)
