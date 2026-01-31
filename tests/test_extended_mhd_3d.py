"""Tests for 3D extended MHD model."""

import jax.numpy as jnp
import pytest
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


class TestExtendedMHD3D:
    """Test 3D extended MHD."""

    def test_model_creation(self):
        """Test creating extended MHD model."""
        model = ExtendedMHD(eta=1e-4, include_hall=True)
        assert model.eta == 1e-4
        assert model.include_hall is True

    def test_compute_rhs_shape(self):
        """Test RHS computation returns correct shapes."""
        model = ExtendedMHD(eta=1e-4)
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
            Te=jnp.ones((8, 8, 8)) * 100 * 1.602e-19,
        )

        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)

    def test_hall_term_present(self):
        """Hall term should contribute when include_hall=True."""
        geom = Geometry(nx=16, ny=16, nz=16)
        state = State.zeros(nx=16, ny=16, nz=16)
        # Create non-uniform B that generates J with components
        # perpendicular to B for non-trivial J x B
        B = jnp.zeros((16, 16, 16, 3))
        x = geom.x_grid
        y = geom.y_grid
        # B_x varies with y, B_y varies with x -> J_z component
        # B_z varies with x,y -> J_x, J_y components
        B = B.at[..., 0].set(jnp.sin(2 * jnp.pi * y))
        B = B.at[..., 1].set(jnp.sin(2 * jnp.pi * x))
        B = B.at[..., 2].set(jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y))
        state = state.replace(
            B=B,
            n=jnp.ones((16, 16, 16)) * 1e19,
            p=jnp.ones((16, 16, 16)) * 1e3,
        )

        model_with_hall = ExtendedMHD(eta=0.0, include_hall=True)
        model_no_hall = ExtendedMHD(eta=0.0, include_hall=False)

        rhs_hall = model_with_hall.compute_rhs(state, geom)
        rhs_no_hall = model_no_hall.compute_rhs(state, geom)

        # With Hall term, results should differ
        assert not jnp.allclose(rhs_hall.B, rhs_no_hall.B)

    def test_apply_constraints_cleans_divergence(self):
        """apply_constraints should reduce div(B) in ExtendedMHD."""
        from jax_frc.operators import divergence_3d
        from jax_frc.solvers.explicit import RK4Solver

        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")

        # Create state with non-zero divergence (sinusoidal has zero mean)
        B = jnp.zeros((16, 16, 16, 3))
        B = B.at[:, :, :, 0].set(jnp.sin(2 * jnp.pi * geom.x_grid))

        state = State(
            B=B,
            E=jnp.zeros((16, 16, 16, 3)),
            n=jnp.ones((16, 16, 16)) * 1e20,
            p=jnp.ones((16, 16, 16)) * 1e3,
            v=jnp.zeros((16, 16, 16, 3)),
        )

        solver = RK4Solver()

        div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
        cleaned_state = solver._apply_constraints(state, geom)
        div_after = jnp.linalg.norm(divergence_3d(cleaned_state.B, geom))

        assert div_after < div_before * 0.2  # At least 5x reduction
