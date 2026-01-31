# tests/test_finite_volume_mhd.py
"""Tests for FiniteVolumeMHD model."""

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.finite_volume_mhd import FiniteVolumeMHD
from jax_frc.solvers.riemann.wave_speeds import hall_signal_speed


def test_finite_volume_mhd_uniform_state_zero_rhs():
    """Uniform state should yield near-zero RHS with periodic boundaries."""
    nx, ny, nz = 4, 1, 4
    geometry = Geometry(nx=nx, ny=ny, nz=nz, bc_x="periodic", bc_y="periodic", bc_z="periodic")

    n = jnp.full((nx, ny, nz), 1.0)
    p = jnp.full((nx, ny, nz), 1.0)
    v = jnp.zeros((nx, ny, nz, 3))
    B = jnp.full((nx, ny, nz, 3), 0.5)
    E = jnp.zeros((nx, ny, nz, 3))

    state = State(B=B, E=E, n=n, p=p, v=v, time=0.0, step=0)
    model = FiniteVolumeMHD(riemann_solver="hlld")

    rhs = model.compute_rhs(state, geometry)

    assert jnp.allclose(rhs.n, 0.0, atol=1e-8)
    assert jnp.allclose(rhs.v, 0.0, atol=1e-8)
    assert jnp.allclose(rhs.p, 0.0, atol=1e-8)
    assert jnp.allclose(rhs.B, 0.0, atol=1e-8)


def test_compute_stable_dt_hall_uses_hall_signal_speed():
    """Hall CFL should use Hall signal speed (fast + whistler)."""
    geometry = Geometry(nx=8, ny=1, nz=8, bc_x="periodic", bc_y="periodic", bc_z="periodic")
    n = jnp.ones((8, 1, 8))
    p = jnp.ones((8, 1, 8))
    v = jnp.zeros((8, 1, 8, 3))
    B = jnp.ones((8, 1, 8, 3))
    state = State(B=B, E=jnp.zeros_like(B), n=n, p=p, v=v, time=0.0, step=0)

    model = FiniteVolumeMHD(riemann_solver="hll", include_hall=True, cfl=0.4)
    dt = model.compute_stable_dt(state, geometry)

    dx_min = min(geometry.dx, geometry.dz)
    v_mag = jnp.sqrt(jnp.sum(v**2, axis=-1))
    cf = hall_signal_speed(
        n, p, B[..., 0], B[..., 1], B[..., 2], dx_min, model.gamma, model.hall_scale
    )
    expected = model.cfl * dx_min / float(jnp.max(cf + v_mag))

    assert jnp.isclose(dt, expected, atol=1e-12)


def test_compute_cleaning_speed_hall_uses_hall_signal_speed():
    """Divergence cleaning speed should use Hall signal speed when enabled."""
    geometry = Geometry(nx=8, ny=1, nz=8, bc_x="periodic", bc_y="periodic", bc_z="periodic")
    n = jnp.ones((8, 1, 8))
    p = jnp.ones((8, 1, 8))
    v = jnp.zeros((8, 1, 8, 3))
    B = jnp.ones((8, 1, 8, 3))
    state = State(B=B, E=jnp.zeros_like(B), n=n, p=p, v=v, time=0.0, step=0)

    model = FiniteVolumeMHD(riemann_solver="hll", include_hall=True)
    ch = model.compute_cleaning_speed(state, geometry)

    dx_min = min(geometry.dx, geometry.dz)
    v_mag = jnp.sqrt(jnp.sum(v**2, axis=-1))
    cf = hall_signal_speed(
        n, p, B[..., 0], B[..., 1], B[..., 2], dx_min, model.gamma, model.hall_scale
    )
    expected = 0.95 * float(jnp.max(cf + v_mag))

    assert jnp.isclose(ch, expected, atol=1e-12)
