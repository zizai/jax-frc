"""Explicit time integration schemes."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax_frc.solvers.base import Solver
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

@dataclass
class EulerSolver(Solver):
    """Simple forward Euler integration.

    Handles both psi-based (Resistive MHD) and B-based (Extended MHD) evolution.
    """

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        rhs = model.compute_rhs(state, geometry)

        # Update psi if RHS has non-zero psi (Resistive MHD)
        new_psi = state.psi + dt * rhs.psi

        # Update B if RHS has non-zero B (Extended MHD)
        new_B = state.B + dt * rhs.B

        # Update E if computed (Hybrid Kinetic)
        new_E = rhs.E if jnp.any(rhs.E != 0) else state.E

        new_state = state.replace(
            psi=new_psi,
            B=new_B,
            E=new_E,
            time=state.time + dt,
            step=state.step + 1
        )
        return model.apply_constraints(new_state, geometry)

@dataclass
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration.

    Handles both psi-based (Resistive MHD) and B-based (Extended MHD) evolution.
    """

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # k1
        k1 = model.compute_rhs(state, geometry)

        # k2
        state_k2 = state.replace(
            psi=state.psi + 0.5*dt*k1.psi,
            B=state.B + 0.5*dt*k1.B
        )
        k2 = model.compute_rhs(state_k2, geometry)

        # k3
        state_k3 = state.replace(
            psi=state.psi + 0.5*dt*k2.psi,
            B=state.B + 0.5*dt*k2.B
        )
        k3 = model.compute_rhs(state_k3, geometry)

        # k4
        state_k4 = state.replace(
            psi=state.psi + dt*k3.psi,
            B=state.B + dt*k3.B
        )
        k4 = model.compute_rhs(state_k4, geometry)

        # Combine
        new_psi = state.psi + (dt/6) * (k1.psi + 2*k2.psi + 2*k3.psi + k4.psi)
        new_B = state.B + (dt/6) * (k1.B + 2*k2.B + 2*k3.B + k4.B)

        # Get E from final RHS (for Hybrid)
        new_E = k4.E if jnp.any(k4.E != 0) else state.E

        new_state = state.replace(
            psi=new_psi,
            B=new_B,
            E=new_E,
            time=state.time + dt,
            step=state.step + 1
        )
        return model.apply_constraints(new_state, geometry)
