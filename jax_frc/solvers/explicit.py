"""Explicit time integration schemes."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax_frc.solvers.base import Solver
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel


@dataclass(frozen=True)
class EulerSolver(Solver):
    """Simple forward Euler integration.

    Updates B field based on model RHS (dB/dt).
    """

    @partial(jax.jit, static_argnums=(0, 3, 4))  # self, model, geometry static
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        rhs = model.compute_rhs(state, geometry)

        # Update MHD fields (n, v, p, B) if they exist
        new_n = state.n + dt * rhs.n if state.n is not None and rhs.n is not None else state.n
        new_v = state.v + dt * rhs.v if state.v is not None and rhs.v is not None else state.v
        new_p = state.p + dt * rhs.p if state.p is not None and rhs.p is not None else state.p
        new_B = state.B + dt * rhs.B
        psi_base = state.psi if state.psi is not None else (
            jnp.zeros_like(rhs.psi) if rhs.psi is not None else None
        )
        new_psi = psi_base + dt * rhs.psi if rhs.psi is not None else state.psi

        # Update E if computed (Hybrid Kinetic) - use lax.cond for JIT compatibility
        new_E = lax.cond(
            jnp.any(rhs.E != 0),
            lambda: rhs.E,
            lambda: state.E
        )

        # Update Te if computed
        new_Te = None
        if state.Te is not None and rhs.Te is not None:
            new_Te = state.Te + dt * rhs.Te

        new_state = state.replace(
            n=new_n,
            v=new_v,
            p=new_p,
            B=new_B,
            psi=new_psi,
            E=new_E,
            Te=new_Te,
            time=state.time + dt,
            step=state.step + 1,
        )
        return new_state


@dataclass(frozen=True)
class RK2Solver(Solver):
    """2nd-order Runge-Kutta (Heun/SSPRK2) integration."""

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        def add_scaled_rhs(base: State, rhs: State, scale: float) -> State:
            new_n = base.n + scale * rhs.n if base.n is not None and rhs.n is not None else base.n
            new_v = base.v + scale * rhs.v if base.v is not None and rhs.v is not None else base.v
            new_p = base.p + scale * rhs.p if base.p is not None and rhs.p is not None else base.p
            new_B = base.B + scale * rhs.B
            psi_base = base.psi if base.psi is not None else (
                jnp.zeros_like(rhs.psi) if rhs.psi is not None else None
            )
            new_psi = psi_base + scale * rhs.psi if rhs.psi is not None else base.psi
            new_E = base.E + scale * rhs.E if rhs.E is not None else base.E
            new_Te = None
            if base.Te is not None and rhs.Te is not None:
                new_Te = base.Te + scale * rhs.Te
            return base.replace(n=new_n, v=new_v, p=new_p, B=new_B, psi=new_psi, E=new_E, Te=new_Te)

        k1 = model.compute_rhs(state, geometry)
        state_k2 = add_scaled_rhs(state, k1, dt)
        k2 = model.compute_rhs(state_k2, geometry)

        new_n = state.n + 0.5 * dt * (k1.n + k2.n) if state.n is not None and k1.n is not None else state.n
        new_v = state.v + 0.5 * dt * (k1.v + k2.v) if state.v is not None and k1.v is not None else state.v
        new_p = state.p + 0.5 * dt * (k1.p + k2.p) if state.p is not None and k1.p is not None else state.p
        new_B = state.B + 0.5 * dt * (k1.B + k2.B)
        psi_base = state.psi if state.psi is not None else (
            jnp.zeros_like(k1.psi) if k1.psi is not None else None
        )
        new_psi = psi_base
        if k1.psi is not None:
            new_psi = psi_base + 0.5 * dt * (k1.psi + k2.psi)

        new_E = state.E
        if k1.E is not None:
            new_E = state.E + 0.5 * dt * (k1.E + k2.E)

        new_Te = None
        if state.Te is not None and k1.Te is not None:
            new_Te = state.Te + 0.5 * dt * (k1.Te + k2.Te)

        new_state = state.replace(
            n=new_n,
            v=new_v,
            p=new_p,
            B=new_B,
            psi=new_psi,
            E=new_E,
            Te=new_Te,
            time=state.time + dt,
            step=state.step + 1,
        )
        return model.apply_constraints(new_state, geometry)


@dataclass(frozen=True)
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration.

    Updates B field and Te field based on model RHS.
    """

    @partial(jax.jit, static_argnums=(0, 3, 4))  # self, model, geometry static
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """RK4 time step updating all state fields."""

        def add_scaled_rhs(base: State, rhs: State, scale: float) -> State:
            """Add scaled RHS to base state for all fields."""
            new_n = base.n + scale * rhs.n if base.n is not None and rhs.n is not None else base.n
            new_v = base.v + scale * rhs.v if base.v is not None and rhs.v is not None else base.v
            new_p = base.p + scale * rhs.p if base.p is not None and rhs.p is not None else base.p
            new_B = base.B + scale * rhs.B
            psi_base = base.psi if base.psi is not None else (
                jnp.zeros_like(rhs.psi) if rhs.psi is not None else None
            )
            new_psi = psi_base + scale * rhs.psi if rhs.psi is not None else base.psi
            new_E = base.E + scale * rhs.E if rhs.E is not None else base.E
            new_Te = None
            if base.Te is not None and rhs.Te is not None:
                new_Te = base.Te + scale * rhs.Te
            return base.replace(n=new_n, v=new_v, p=new_p, B=new_B, psi=new_psi, E=new_E, Te=new_Te)

        # k1
        k1 = model.compute_rhs(state, geometry)

        # k2
        state_k2 = add_scaled_rhs(state, k1, 0.5 * dt)
        k2 = model.compute_rhs(state_k2, geometry)

        # k3
        state_k3 = add_scaled_rhs(state, k2, 0.5 * dt)
        k3 = model.compute_rhs(state_k3, geometry)

        # k4
        state_k4 = add_scaled_rhs(state, k3, dt)
        k4 = model.compute_rhs(state_k4, geometry)

        # Combine: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        new_n = state.n + (dt/6) * (k1.n + 2*k2.n + 2*k3.n + k4.n) if state.n is not None and k1.n is not None else state.n
        new_v = state.v + (dt/6) * (k1.v + 2*k2.v + 2*k3.v + k4.v) if state.v is not None and k1.v is not None else state.v
        new_p = state.p + (dt/6) * (k1.p + 2*k2.p + 2*k3.p + k4.p) if state.p is not None and k1.p is not None else state.p
        new_B = state.B + (dt/6) * (k1.B + 2*k2.B + 2*k3.B + k4.B)
        psi_base = state.psi if state.psi is not None else (
            jnp.zeros_like(k1.psi) if k1.psi is not None else None
        )
        new_psi = psi_base
        if k1.psi is not None:
            new_psi = psi_base + (dt/6) * (k1.psi + 2*k2.psi + 2*k3.psi + k4.psi)

        new_E = state.E
        if k1.E is not None:
            new_E = state.E + (dt/6) * (k1.E + 2*k2.E + 2*k3.E + k4.E)

        new_Te = None
        if state.Te is not None and k1.Te is not None:
            new_Te = state.Te + (dt/6) * (k1.Te + 2*k2.Te + 2*k3.Te + k4.Te)

        new_state = state.replace(
            n=new_n,
            v=new_v,
            p=new_p,
            B=new_B,
            psi=new_psi,
            E=new_E,
            Te=new_Te,
            time=state.time + dt,
            step=state.step + 1,
        )
        return new_state


@dataclass(frozen=True)
class SemiLagrangianSolver(Solver):
    """Semi-Lagrangian solver for advection-dominated problems.

    Uses backward characteristic tracing to eliminate numerical diffusion
    from the advection term. Ideal for high magnetic Reynolds number (Rm >> 1)
    flows where advection dominates over diffusion.

    The advection term is handled by tracing characteristics backward:
        B(x, t+dt) = B(x - v*dt, t)

    The diffusion term (if any) is handled with standard finite differences.
    """

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Semi-Lagrangian time step.

        1. Advect B using backward characteristic tracing
        2. Add diffusion term using finite differences
        """
        from jax_frc.solvers.constrained_transport import advect_semi_lagrangian
        from jax_frc.operators import laplacian_3d
        from jax_frc.constants import MU0

        B = state.B
        v = state.v

        # Step 1: Advect B using semi-Lagrangian method
        if v is not None:
            B_advected = advect_semi_lagrangian(B, v, geometry, dt)
        else:
            B_advected = B

        # Step 2: Add diffusion term if model has resistivity
        eta = getattr(model, 'eta', 0.0)
        if eta > 0:
            # Diffusion: dB/dt = eta/mu0 * laplacian(B)
            diffusion_term = eta / MU0 * laplacian_3d(B_advected, geometry)
            new_B = B_advected + dt * diffusion_term
        else:
            new_B = B_advected

        new_state = state.replace(
            B=new_B,
            time=state.time + dt,
            step=state.step + 1,
        )
        return new_state
