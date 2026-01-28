"""Tests for constrained transport boundary handling."""

import jax.numpy as jnp

from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d
from jax_frc.solvers.constrained_transport import compute_emf_upwind, induction_rhs_ct


def _extrapolate_boundary(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    left_slice = [slice(None)] * field.ndim
    left_slice[axis] = 0
    left_src = [slice(None)] * field.ndim
    left_src[axis] = 1
    right_slice = [slice(None)] * field.ndim
    right_slice[axis] = -1
    right_src = [slice(None)] * field.ndim
    right_src[axis] = -2
    field = field.at[tuple(left_slice)].set(field[tuple(left_src)])
    field = field.at[tuple(right_slice)].set(field[tuple(right_src)])
    return field


def _apply_boundary_E(E: jnp.ndarray, geometry: Geometry) -> jnp.ndarray:
    for axis, bc in enumerate([geometry.bc_x, geometry.bc_y, geometry.bc_z]):
        if bc != "periodic":
            E = _extrapolate_boundary(E, axis)
    return E


def test_ct_boundary_e_field_neumann():
    """Neumann BCs should avoid periodic wrap in CT curl."""
    geom = Geometry(
        nx=8, ny=8, nz=8,
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0,
        bc_x="neumann", bc_y="neumann", bc_z="neumann"
    )
    L = geom.x_max - geom.x_min
    v = jnp.zeros((8, 8, 8, 3))
    v = v.at[..., 1].set(jnp.cos(jnp.pi * geom.x_grid / L))
    B = jnp.zeros((8, 8, 8, 3))
    B = B.at[..., 0].set(1.0)

    dB = induction_rhs_ct(v, B, geom)
    E = compute_emf_upwind(v, B, geom)
    E_bc = _apply_boundary_E(E, geom)
    expected = -curl_3d(E_bc, geom, order=4)
    assert jnp.allclose(dB, expected, atol=1e-6)
