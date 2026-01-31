"""Consistency tests for MHD RHS and solver updates."""
from __future__ import annotations

import jax.numpy as jnp
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import RK4Solver
from tests.utils.cartesian import make_geometry
from tests.utils.residuals import max_abs, relative_l2_norm


def make_uniform_state(geometry: Geometry) -> State:
    """Build a uniform state that should yield zero RHS."""
    n = jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * 1e19
    p = jnp.ones_like(n) * 1e3
    v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    E = jnp.zeros_like(B)
    return State(B=B, E=E, n=n, p=p, v=v)


def make_diffusion_state(geometry: Geometry) -> State:
    """Build a state with a non-uniform B field for diffusion tests."""
    state = make_uniform_state(geometry)
    length = geometry.x_max - geometry.x_min
    Bz = jnp.sin(2 * jnp.pi * geometry.x_grid / length)
    B = state.B.at[..., 2].set(0.1 * Bz)
    return state.replace(B=B)


def test_resistive_mhd_uniform_equilibrium_rhs_zero():
    """Uniform fields should produce a near-zero RHS in ResistiveMHD."""
    geometry = make_geometry(nx=12, ny=1, nz=12, extent=1.0)
    model = ResistiveMHD(
        eta=1e-4,
        advection_scheme="ct",
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    state = make_uniform_state(geometry)

    rhs = model.compute_rhs(state, geometry)

    assert max_abs(rhs.n) < 1e-10
    assert max_abs(rhs.p) < 1e-10
    assert max_abs(rhs.v) < 1e-10
    assert max_abs(rhs.B) < 1e-10


def test_extended_mhd_uniform_equilibrium_rhs_zero():
    """Uniform fields should produce a near-zero RHS in ExtendedMHD."""
    geometry = make_geometry(nx=12, ny=1, nz=12, extent=1.0)
    model = ExtendedMHD(
        eta=1e-4,
        include_hall=True,
        include_electron_pressure=True,
        apply_divergence_cleaning=False,
    )
    state = make_uniform_state(geometry)

    rhs = model.compute_rhs(state, geometry)

    assert max_abs(rhs.n) < 1e-10
    assert max_abs(rhs.p) < 1e-10
    assert max_abs(rhs.v) < 1e-10
    assert max_abs(rhs.B) < 1e-10
    assert rhs.Te is None


def test_resistive_mhd_rk4_tiny_dt_rhs_consistency():
    """Tiny dt RK4 update should match RHS for B evolution."""
    geometry = make_geometry(nx=16, ny=1, nz=16, extent=1.0)
    model = ResistiveMHD(
        eta=1e-4,
        advection_scheme="ct",
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    solver = RK4Solver()
    state = make_diffusion_state(geometry)

    dt = 1e-7
    rhs = model.compute_rhs(state, geometry)
    next_state = solver.step(state, dt, model, geometry)

    residual = relative_l2_norm((next_state.B - state.B) / dt, rhs.B)
    assert residual < 5e-4


def test_extended_mhd_rk4_tiny_dt_rhs_consistency():
    """Tiny dt RK4 update should match RHS for ExtendedMHD B evolution."""
    geometry = make_geometry(nx=16, ny=1, nz=16, extent=1.0)
    model = ExtendedMHD(
        eta=1e-4,
        include_hall=False,
        include_electron_pressure=False,
        apply_divergence_cleaning=False,
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    solver = RK4Solver()
    state = make_diffusion_state(geometry)

    dt = 1e-7
    rhs = model.compute_rhs(state, geometry)
    next_state = solver.step(state, dt, model, geometry)

    residual = relative_l2_norm((next_state.B - state.B) / dt, rhs.B)
    assert residual < 5e-4
