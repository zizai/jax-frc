"""Tests for JIT compilation of physics models and solvers."""

import pytest
import jax
import jax.numpy as jnp
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestResistiveMHDJIT:
    """Test JIT compilation of ResistiveMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        state = State.zeros(geometry.nr, geometry.nz)
        # Add some non-trivial psi
        r, z = geometry.r_grid, geometry.z_grid
        state = state.replace(psi=jnp.exp(-r**2 - z**2))
        model = ResistiveMHD(resistivity=SpitzerResistivity())
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        # This should not raise
        # Note: static_argnums=(1,) because it's a bound method (self is bound)
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result = jitted_rhs(state, geometry)
        assert result.psi.shape == state.psi.shape

    def test_compute_rhs_jit_produces_same_result(self, setup):
        """JIT and non-JIT produce identical results."""
        model, state, geometry = setup
        result_eager = model.compute_rhs(state, geometry)
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result_jit = jitted_rhs(state, geometry)
        assert jnp.allclose(result_eager.psi, result_jit.psi, rtol=1e-5)


from jax_frc.models.extended_mhd import HaloDensityModel


class TestExtendedMHDJIT:
    """Test JIT compilation of ExtendedMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        state = State.zeros(geometry.nr, geometry.nz)
        r, z = geometry.r_grid, geometry.z_grid
        # Initialize B field
        state = state.replace(
            B=jnp.stack([jnp.zeros_like(r), jnp.zeros_like(r), jnp.ones_like(r)], axis=-1),
            n=jnp.ones_like(r) * 1e19,
        )
        model = ExtendedMHD(resistivity=SpitzerResistivity(), halo_model=HaloDensityModel())
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result = jitted_rhs(state, geometry)
        assert result.B.shape == state.B.shape
