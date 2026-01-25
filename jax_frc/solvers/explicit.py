"""Explicit time integration schemes."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax_frc.solvers.base import Solver
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

@dataclass
class EulerSolver(Solver):
    """Simple forward Euler integration."""

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        rhs = model.compute_rhs(state, geometry)
        new_psi = state.psi + dt * rhs.psi
        new_state = state.replace(psi=new_psi, time=state.time + dt, step=state.step + 1)
        return model.apply_constraints(new_state, geometry)

@dataclass
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration."""

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # k1
        k1 = model.compute_rhs(state, geometry)

        # k2
        state_k2 = state.replace(psi=state.psi + 0.5*dt*k1.psi)
        k2 = model.compute_rhs(state_k2, geometry)

        # k3
        state_k3 = state.replace(psi=state.psi + 0.5*dt*k2.psi)
        k3 = model.compute_rhs(state_k3, geometry)

        # k4
        state_k4 = state.replace(psi=state.psi + dt*k3.psi)
        k4 = model.compute_rhs(state_k4, geometry)

        # Combine
        new_psi = state.psi + (dt/6) * (k1.psi + 2*k2.psi + 2*k3.psi + k4.psi)
        new_state = state.replace(psi=new_psi, time=state.time + dt, step=state.step + 1)
        return model.apply_constraints(new_state, geometry)
