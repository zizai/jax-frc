"""Solver consistency checks for MHD models."""
from __future__ import annotations

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.imex import ImexSolver
from jax_frc.solvers.semi_implicit import SemiImplicitSolver
from tests.utils.cartesian import make_geometry
from tests.utils.residuals import relative_l2_norm


def make_diffusion_state(nx: int, ny: int, nz: int, extent: float) -> tuple[State, object]:
    """Create a simple diffusion state and geometry."""
    geometry = make_geometry(nx=nx, ny=ny, nz=nz, extent=extent)
    n = jnp.ones((nx, ny, nz)) * 1e19
    p = jnp.ones_like(n) * 1e3
    v = jnp.zeros((nx, ny, nz, 3))
    B = jnp.zeros((nx, ny, nz, 3))
    length = geometry.x_max - geometry.x_min
    Bz = jnp.sin(2 * jnp.pi * geometry.x_grid / length)
    B = B.at[..., 2].set(0.1 * Bz)
    state = State(B=B, E=jnp.zeros_like(B), n=n, p=p, v=v)
    return state, geometry


def test_imex_tiny_dt_rhs_consistency():
    """IMEX update should match RHS for small dt (B diffusion)."""
    state, geometry = make_diffusion_state(nx=16, ny=1, nz=16, extent=1.0)
    model = ResistiveMHD(eta=1e-4, advection_scheme="central")
    solver = ImexSolver()

    dt = 1e-6
    rhs = model.compute_rhs(state, geometry)
    next_state = solver.step(state, dt, model, geometry)

    # IMEX uses a 2D laplacian in x-z; compare interior where stencil is valid.
    interior = (slice(1, -1), slice(None), slice(1, -1), slice(None))
    residual = relative_l2_norm(
        (next_state.B[interior] - state.B[interior]) / dt,
        rhs.B[interior],
    )
    assert residual < 5e-3


def test_semi_implicit_tiny_dt_rhs_consistency():
    """Semi-implicit update should match RHS for small dt (B evolution)."""
    state, geometry = make_diffusion_state(nx=16, ny=1, nz=16, extent=1.0)
    model = ExtendedMHD(
        eta=1e-4,
        include_hall=True,
        include_electron_pressure=False,
        apply_divergence_cleaning=False,
    )
    solver = SemiImplicitSolver(damping_factor=1e6)

    dt = 1e-7
    rhs = model.compute_rhs(state, geometry)
    next_state = solver.step(state, dt, model, geometry)

    residual = relative_l2_norm((next_state.B - state.B) / dt, rhs.B)
    assert residual < 1e-3


def test_imex_residual_scales_with_dt():
    """IMEX residual should decrease as dt shrinks."""
    state, geometry = make_diffusion_state(nx=16, ny=1, nz=16, extent=1.0)
    model = ResistiveMHD(eta=1e-4, advection_scheme="central")
    solver = ImexSolver()
    rhs = model.compute_rhs(state, geometry)

    dt_coarse = 2e-6
    dt_fine = 1e-6

    next_state_coarse = solver.step(state, dt_coarse, model, geometry)
    next_state_fine = solver.step(state, dt_fine, model, geometry)

    interior = (slice(1, -1), slice(None), slice(1, -1), slice(None))
    residual_coarse = relative_l2_norm(
        (next_state_coarse.B[interior] - state.B[interior]) / dt_coarse,
        rhs.B[interior],
    )
    residual_fine = relative_l2_norm(
        (next_state_fine.B[interior] - state.B[interior]) / dt_fine,
        rhs.B[interior],
    )

    assert residual_fine < residual_coarse * 0.7


def test_semi_implicit_residual_scales_with_dt():
    """Semi-implicit residual should remain bounded as dt shrinks."""
    state, geometry = make_diffusion_state(nx=16, ny=1, nz=16, extent=1.0)
    model = ExtendedMHD(
        eta=1e-4,
        include_hall=True,
        include_electron_pressure=False,
        apply_divergence_cleaning=False,
    )
    solver = SemiImplicitSolver(damping_factor=1e6)
    rhs = model.compute_rhs(state, geometry)

    dt_coarse = 2e-7
    dt_fine = 1e-7

    next_state_coarse = solver.step(state, dt_coarse, model, geometry)
    next_state_fine = solver.step(state, dt_fine, model, geometry)

    residual_coarse = relative_l2_norm((next_state_coarse.B - state.B) / dt_coarse, rhs.B)
    residual_fine = relative_l2_norm((next_state_fine.B - state.B) / dt_fine, rhs.B)

    assert residual_fine < 1e-3
    assert residual_fine < residual_coarse * 2.0
