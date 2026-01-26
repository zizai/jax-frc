"""Tests for ResistiveMHD SplitRHS interface."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity


def test_explicit_rhs_returns_advection_only():
    """explicit_rhs returns only advection term, no diffusion."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    v = jnp.zeros((16, 16, 3))
    v = v.at[:, :, 0].set(0.1)  # v_r = 0.1 m/s
    state = state.replace(psi=psi, v=v)

    rhs = model.explicit_rhs(state, geometry, t=0.0)
    assert jnp.any(rhs.psi != 0)


def test_implicit_rhs_returns_diffusion_only():
    """implicit_rhs returns only diffusion term."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    state = state.replace(psi=psi)

    rhs = model.implicit_rhs(state, geometry, t=0.0)
    assert jnp.any(rhs.psi != 0)


def test_apply_implicit_operator():
    """apply_implicit_operator applies (I - theta*dt*L)."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.ones((16, 16))
    state = state.replace(psi=psi)

    result = model.apply_implicit_operator(state, geometry, dt=1e-6, theta=1.0)
    assert result.psi.shape == psi.shape


def test_explicit_plus_implicit_equals_compute_rhs():
    """Sum of explicit and implicit RHS should equal compute_rhs."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    v = jnp.zeros((16, 16, 3))
    v = v.at[:, :, 0].set(0.1)
    state = state.replace(psi=psi, v=v)

    # Get individual terms
    explicit = model.explicit_rhs(state, geometry)
    implicit = model.implicit_rhs(state, geometry)
    combined = model.compute_rhs(state, geometry)

    # Sum should equal combined (within tolerance)
    sum_rhs = explicit.psi + implicit.psi
    assert jnp.allclose(sum_rhs, combined.psi, rtol=1e-10)


def test_explicit_rhs_zero_velocity_gives_zero():
    """explicit_rhs with zero velocity should return zero advection."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5)
    state = state.replace(psi=psi)  # v is already zero

    rhs = model.explicit_rhs(state, geometry)
    assert jnp.allclose(rhs.psi, 0.0)


def test_apply_implicit_operator_identity_with_zero_dt():
    """apply_implicit_operator with dt=0 should return identity."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5)
    state = state.replace(psi=psi)

    result = model.apply_implicit_operator(state, geometry, dt=0.0, theta=1.0)
    assert jnp.allclose(result.psi, psi)
