"""Tests for Dedner/GLM divergence cleaning utilities."""

import jax.numpy as jnp

from jax_frc.core.geometry import Geometry
from jax_frc.solvers.riemann.dedner import glm_cleaning_update
from jax_frc.solvers.riemann.hll_full import hll_update_full_with_dedner
from jax_frc.solvers.riemann.mhd_state import primitive_to_conserved, MHDPrimitive


def test_glm_cleaning_respects_neumann_bc():
    """Neumann BC should avoid periodic wrap in cleaning flux."""
    geom = Geometry(nx=6, ny=1, nz=4, bc_x="neumann", bc_y="periodic", bc_z="periodic")
    B = jnp.zeros((6, 1, 4, 3))
    B = B.at[-1, :, :, 0].set(1.0)  # sharp jump at right boundary in Bx
    psi = jnp.zeros((6, 1, 4))

    dB_clean, _ = glm_cleaning_update(B, geom, psi, ch=1.0, beta=1.3)

    # With Neumann BC, left boundary should not feel right boundary jump
    assert jnp.allclose(dB_clean[0, :, :, 0], 0.0, atol=1e-8)


def test_hll_update_full_with_dedner_matches_glm_for_bn_flux():
    """Dedner HLL update should match GLM cleaning for Bn/psi on quiescent state."""
    geom = Geometry(nx=6, ny=1, nz=4, bc_x="periodic", bc_y="periodic", bc_z="periodic")
    B = jnp.zeros((6, 1, 4, 3))
    B = B.at[:, :, :, 0].set(jnp.linspace(0.0, 1.0, 6)[:, None, None])
    psi = jnp.zeros((6, 1, 4))

    prim = MHDPrimitive(
        rho=jnp.ones((6, 1, 4)),
        vx=jnp.zeros((6, 1, 4)),
        vy=jnp.zeros((6, 1, 4)),
        vz=jnp.zeros((6, 1, 4)),
        p=jnp.ones((6, 1, 4)),
        Bx=B[..., 0],
        By=B[..., 1],
        Bz=B[..., 2],
    )
    cons = primitive_to_conserved(prim, gamma=5.0 / 3.0)

    dU, dpsi = hll_update_full_with_dedner(
        cons, psi, geom, gamma=5.0 / 3.0, beta=1.3, ch=1.0
    )
    dB_clean, dpsi_clean = glm_cleaning_update(B, geom, psi, ch=1.0, beta=1.3)

    assert jnp.allclose(dU.Bx, dB_clean[..., 0], atol=1e-10)
    assert jnp.allclose(dU.Bz, dB_clean[..., 2], atol=1e-10)
    assert jnp.allclose(dpsi, dpsi_clean, atol=1e-10)
