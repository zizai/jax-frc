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
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        rhs = model.compute_rhs(state, geometry)

        # Update B from dB/dt
        new_B = state.B + dt * rhs.B

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
            B=new_B,
            E=new_E,
            Te=new_Te,
            time=state.time + dt,
            step=state.step + 1,
        )
        return model.apply_constraints(new_state, geometry)

@dataclass(frozen=True)
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration.

    Updates B field based on model RHS (dB/dt).
    """

    @partial(jax.jit, static_argnums=(0, 3, 4))  # self, model, geometry static
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # k1
        k1 = model.compute_rhs(state, geometry)

        # k2
        state_k2 = state.replace(B=state.B + 0.5*dt*k1.B)
        k2 = model.compute_rhs(state_k2, geometry)

        # k3
        state_k3 = state.replace(B=state.B + 0.5*dt*k2.B)
        k3 = model.compute_rhs(state_k3, geometry)

        # k4
        state_k4 = state.replace(B=state.B + dt*k3.B)
        k4 = model.compute_rhs(state_k4, geometry)

        # Combine
        new_B = state.B + (dt/6) * (k1.B + 2*k2.B + 2*k3.B + k4.B)

        # Get E from final RHS (for Hybrid) - use lax.cond for JIT compatibility
        new_E = lax.cond(
            jnp.any(k4.E != 0),
            lambda: k4.E,
            lambda: state.E
        )

        new_state = state.replace(
            B=new_B,
            E=new_E,
            time=state.time + dt,
            step=state.step + 1,
        )
        return model.apply_constraints(new_state, geometry)
