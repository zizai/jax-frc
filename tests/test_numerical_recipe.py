"""Tests for NumericalRecipe stepping and divergence handling."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel
from jax_frc.operators import divergence_3d
from jax_frc.solvers import RK4Solver, TimeController
from jax_frc.solvers.recipe import NumericalRecipe
from tests.utils.cartesian import make_geometry


@dataclass(frozen=True)
class DummyModel(PhysicsModel):
    """Minimal physics model for testing recipe stepping behavior."""

    def compute_rhs(self, state, geometry):
        zeros_B = jnp.zeros_like(state.B)
        zeros_E = jnp.zeros_like(state.E)
        zeros_n = jnp.zeros_like(state.n)
        zeros_p = jnp.zeros_like(state.p)
        zeros_psi = jnp.zeros_like(state.psi) if state.psi is not None else None
        return state.replace(B=zeros_B, E=zeros_E, n=zeros_n, p=zeros_p, psi=zeros_psi)

    def compute_stable_dt(self, state, geometry):
        return 0.2

    def apply_constraints(self, state, geometry):
        return state.replace(p=state.p + 1.0)


def test_recipe_step_uses_time_controller_and_constraints_once():
    geom = make_geometry(nx=4, ny=2, nz=4)
    model = DummyModel()
    solver = RK4Solver()
    tc = TimeController(cfl_safety=1.0, dt_min=0.0, dt_max=1.0)
    recipe = NumericalRecipe(solver=solver, time_controller=tc, divergence_strategy="none")

    state = State.zeros(geom.nx, geom.ny, geom.nz)
    next_state = recipe.step(state, model, geom)

    assert next_state.step == 1
    assert next_state.time == pytest.approx(0.2)
    assert jnp.allclose(next_state.p, state.p + 1.0)


def test_recipe_divergence_cleaning_reduces_div_b():
    geom = make_geometry(nx=8, ny=2, nz=8)
    model = DummyModel()
    solver = RK4Solver()
    tc = TimeController(cfl_safety=1.0, dt_min=0.0, dt_max=1.0)
    recipe = NumericalRecipe(solver=solver, time_controller=tc, divergence_strategy="clean")

    state = State.zeros(geom.nx, geom.ny, geom.nz)
    B = state.B.at[..., 0].set(geom.x_grid)
    state = state.replace(B=B)

    div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
    next_state = recipe.step(state, model, geom)
    div_after = jnp.linalg.norm(divergence_3d(next_state.B, geom))

    assert div_after <= div_before
