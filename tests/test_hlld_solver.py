# tests/test_hlld_solver.py
"""Tests for HLLD solver and MHD wave speeds."""

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.riemann.mhd_state import (
    MHDPrimitive,
    primitive_to_conserved,
    compute_mhd_flux,
)
from jax_frc.solvers.riemann.hlld import hlld_flux_1d
from jax_frc.solvers.riemann.wave_speeds import mhd_wave_speeds


def _make_uniform_primitive(shape, rho=1.0, p=1.0, v=(0.1, -0.2, 0.3), B=(0.4, 0.5, -0.6)):
    """Helper to build uniform primitive state."""
    rho_arr = jnp.full(shape, rho)
    p_arr = jnp.full(shape, p)
    vx_arr = jnp.full(shape, v[0])
    vy_arr = jnp.full(shape, v[1])
    vz_arr = jnp.full(shape, v[2])
    Bx_arr = jnp.full(shape, B[0])
    By_arr = jnp.full(shape, B[1])
    Bz_arr = jnp.full(shape, B[2])
    return MHDPrimitive(
        rho=rho_arr,
        vx=vx_arr,
        vy=vy_arr,
        vz=vz_arr,
        p=p_arr,
        Bx=Bx_arr,
        By=By_arr,
        Bz=Bz_arr,
    )


def _assert_conserved_allclose(lhs, rhs, atol=1e-8, rtol=1e-6):
    """Check all components of conserved tuple are close."""
    assert jnp.allclose(lhs.rho, rhs.rho, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.mom_x, rhs.mom_x, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.mom_y, rhs.mom_y, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.mom_z, rhs.mom_z, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.E, rhs.E, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.Bx, rhs.Bx, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.By, rhs.By, atol=atol, rtol=rtol)
    assert jnp.allclose(lhs.Bz, rhs.Bz, atol=atol, rtol=rtol)


def test_hlld_flux_matches_physical_flux_for_equal_states():
    """HLLD flux should reduce to physical flux when UL == UR."""
    shape = (1, 1, 1)
    prim = _make_uniform_primitive(shape)
    cons = primitive_to_conserved(prim)

    for direction in (0, 1, 2):
        flux = compute_mhd_flux(prim, cons, direction)
        hlld_flux = hlld_flux_1d(
            prim, prim, cons, cons, flux, flux, direction, 5.0 / 3.0
        )
        _assert_conserved_allclose(hlld_flux, flux)


def test_hlld_flux_finite_when_normal_field_zero():
    """HLLD should remain finite when Bn == 0 (degenerate case)."""
    shape = (1, 1, 1)
    prim_L = _make_uniform_primitive(shape, B=(0.0, 0.3, -0.2), v=(0.1, 0.2, -0.1))
    prim_R = _make_uniform_primitive(shape, B=(0.0, -0.1, 0.4), v=(-0.2, 0.1, 0.0))

    cons_L = primitive_to_conserved(prim_L)
    cons_R = primitive_to_conserved(prim_R)
    flux_L = compute_mhd_flux(prim_L, cons_L, 0)
    flux_R = compute_mhd_flux(prim_R, cons_R, 0)

    flux = hlld_flux_1d(prim_L, prim_R, cons_L, cons_R, flux_L, flux_R, 0, 5.0 / 3.0)

    for field in flux:
        assert jnp.all(jnp.isfinite(field))


def test_mhd_wave_speeds_hydro_limit():
    """With B=0, fast speed should equal sound speed and Alfven/slow should be 0."""
    rho = jnp.array(1.0)
    p = jnp.array(1.0)
    Bx = jnp.array(0.0)
    By = jnp.array(0.0)
    Bz = jnp.array(0.0)
    gamma = 5.0 / 3.0

    c_fast, c_alfven, c_slow = mhd_wave_speeds(rho, p, Bx, By, Bz, gamma)

    sound = jnp.sqrt(gamma * p / rho)
    assert jnp.allclose(c_fast, sound, atol=1e-8)
    assert jnp.allclose(c_alfven, 0.0, atol=1e-8)
    assert jnp.allclose(c_slow, 0.0, atol=1e-8)


def test_mhd_wave_speeds_ordering():
    """Wave speeds should be non-negative and fast >= slow."""
    rho = jnp.array(1.0)
    p = jnp.array(0.2)
    Bx = jnp.array(1.5)
    By = jnp.array(0.3)
    Bz = jnp.array(0.4)

    c_fast, c_alfven, c_slow = mhd_wave_speeds(rho, p, Bx, By, Bz)

    assert c_fast >= 0.0
    assert c_alfven >= 0.0
    assert c_slow >= 0.0
    assert c_fast >= c_slow
