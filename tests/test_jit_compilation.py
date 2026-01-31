"""Tests for JIT compilation of physics models and solvers."""

import pytest
import jax
import jax.numpy as jnp

from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import EulerSolver, RK4Solver
from jax_frc.core.state import State
from tests.utils.cartesian import make_geometry


class TestResistiveMHDJIT:
    """Test JIT compilation of ResistiveMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = make_geometry(nx=16, ny=4, nz=32)
        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
        x, z = geometry.x_grid, geometry.z_grid
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(jnp.exp(-x**2 - z**2))
        state = state.replace(B=B)
        model = ResistiveMHD(eta=1e-4)
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        # This should not raise
        # Note: static_argnums=(1,) because it's a bound method (self is bound)
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result = jitted_rhs(state, geometry)
        assert result.B.shape == state.B.shape

    def test_compute_rhs_jit_produces_same_result(self, setup):
        """JIT and non-JIT produce identical results."""
        model, state, geometry = setup
        result_eager = model.compute_rhs(state, geometry)
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result_jit = jitted_rhs(state, geometry)
        assert jnp.allclose(result_eager.B, result_jit.B, rtol=1e-5)


class TestExtendedMHDJIT:
    """Test JIT compilation of ExtendedMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = make_geometry(nx=16, ny=4, nz=32)
        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(1.0)
        state = state.replace(
            B=B,
            n=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * 1e19,
            Te=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * 100.0,
        )
        model = ExtendedMHD(
            eta=1e-4,
            include_hall=False,
            include_electron_pressure=False,
            kappa_perp=1e-2,
        )
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(1,))
        result = jitted_rhs(state, geometry)
        assert result.B.shape == state.B.shape


class TestExplicitSolversJIT:
    """Test JIT compilation of explicit solvers."""

    @pytest.fixture
    def setup(self):
        """Create solver, model, state, geometry."""
        geometry = make_geometry(nx=16, ny=4, nz=32)
        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
        x, z = geometry.x_grid, geometry.z_grid
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(jnp.exp(-x**2 - z**2))
        state = state.replace(B=B)
        model = ResistiveMHD(eta=1e-4)
        return state, model, geometry

    def test_euler_step_is_jittable(self, setup):
        """EulerSolver.step can be JIT-compiled."""
        state, model, geometry = setup
        solver = EulerSolver()
        dt = 1e-6
        # Should compile and run
        result = solver.step_with_dt(state, dt, model, geometry)
        assert result.B.shape == state.B.shape

    def test_rk4_step_is_jittable(self, setup):
        """RK4Solver.step can be JIT-compiled."""
        state, model, geometry = setup
        solver = RK4Solver()
        dt = 1e-6
        result = solver.step_with_dt(state, dt, model, geometry)
        assert result.B.shape == state.B.shape
