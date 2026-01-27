"""Tests for ResistiveMHD compute_rhs behavior."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from tests.utils.cartesian import make_geometry


def test_compute_rhs_nonzero_for_nonuniform_b():
    """Non-uniform B should diffuse (non-zero dB/dt)."""
    geometry = make_geometry(nx=16, ny=1, nz=16, extent=0.5)
    model = ResistiveMHD(eta=1e-6)

    state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B = B.at[:, :, :, 2].set(jnp.sin(jnp.pi * geometry.x_grid / 0.5))
    state = state.replace(B=B)

    rhs = model.compute_rhs(state, geometry)
    assert jnp.any(jnp.abs(rhs.B) > 0)


def test_compute_rhs_zero_for_uniform_b():
    """Uniform B should have near-zero diffusion."""
    geometry = make_geometry(nx=16, ny=1, nz=16, extent=0.5)
    model = ResistiveMHD(eta=1e-6)

    state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B = B.at[:, :, :, 2].set(0.1)
    state = state.replace(B=B)

    rhs = model.compute_rhs(state, geometry)
    assert jnp.allclose(rhs.B, 0.0, atol=1e-6)


def test_compute_stable_dt_positive():
    """Stable timestep should be positive and finite."""
    geometry = make_geometry(nx=8, ny=1, nz=8, extent=0.5)
    model = ResistiveMHD(eta=1e-6)
    state = State.zeros(geometry.nx, geometry.ny, geometry.nz)

    dt = model.compute_stable_dt(state, geometry)
    assert jnp.isfinite(dt)
    assert dt > 0
