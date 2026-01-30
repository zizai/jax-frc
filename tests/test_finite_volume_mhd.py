# tests/test_finite_volume_mhd.py
"""Tests for FiniteVolumeMHD model."""

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.finite_volume_mhd import FiniteVolumeMHD


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
