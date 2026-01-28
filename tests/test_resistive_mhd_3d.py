"""Tests for 3D resistive MHD model."""

import jax.numpy as jnp
import pytest
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


class TestResistiveMHD3D:
    """Test 3D resistive MHD."""

    def test_model_creation(self):
        """Test creating resistive MHD model."""
        model = ResistiveMHD(eta=1e-4)
        assert model.eta == 1e-4

    def test_compute_rhs_shape(self):
        """Test RHS computation returns correct shapes."""
        model = ResistiveMHD(eta=1e-4)
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        # Set non-zero B field
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(1.0)  # Uniform Bz
        state = state.replace(B=B, n=jnp.ones((8, 8, 8)) * 1e19)

        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)

    def test_uniform_field_no_change(self):
        """Uniform B field should have zero dB/dt (no current)."""
        model = ResistiveMHD(eta=1e-4)
        geom = Geometry(nx=16, ny=16, nz=16)
        state = State.zeros(nx=16, ny=16, nz=16)
        B = jnp.zeros((16, 16, 16, 3))
        B = B.at[..., 2].set(1.0)  # Uniform Bz
        state = state.replace(B=B, n=jnp.ones((16, 16, 16)) * 1e19)

        rhs = model.compute_rhs(state, geom)
        # Uniform field => J = curl(B)/mu0 = 0 => dB/dt = 0
        assert jnp.allclose(rhs.B, 0.0, atol=1e-10)

    def test_apply_constraints_cleans_divergence(self):
        """apply_constraints should reduce div(B)."""
        from jax_frc.operators import divergence_3d

        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")

        # Create state with non-zero divergence (sinusoidal has zero mean)
        B = jnp.zeros((16, 16, 16, 3))
        # Bx = sin(2*pi*x) has div(B) = 2*pi*cos(2*pi*x) which has zero mean
        B = B.at[:, :, :, 0].set(jnp.sin(2 * jnp.pi * geom.x_grid))

        state = State(
            B=B,
            E=jnp.zeros((16, 16, 16, 3)),
            n=jnp.ones((16, 16, 16)),
            p=jnp.ones((16, 16, 16)),
            v=jnp.zeros((16, 16, 16, 3)),
        )

        model = ResistiveMHD(eta=1e-4)

        div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
        cleaned_state = model.apply_constraints(state, geom)
        div_after = jnp.linalg.norm(divergence_3d(cleaned_state.B, geom))

        assert div_after < div_before * 0.2  # At least 5x reduction
