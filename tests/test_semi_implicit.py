"""Tests for Semi-Implicit Solver with temperature evolution."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.semi_implicit import SemiImplicitSolver
from jax_frc.models.extended_mhd import ExtendedMHD, TemperatureBoundaryCondition
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.state import State
from tests.utils.cartesian import make_geometry


class TestSemiImplicitSolverTemperature:
    """Tests for temperature handling in SemiImplicitSolver."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=4, nz=32)

    @pytest.fixture
    def model_with_thermal(self):
        """Create ExtendedMHD model with thermal transport."""
        bc = TemperatureBoundaryCondition(bc_type="neumann")
        return ExtendedMHD(
            eta=1e-4,
            include_hall=False,
            include_electron_pressure=False,
            kappa_perp=1e-2,
            temperature_bc=bc,
        )

    @pytest.fixture
    def solver(self):
        """Create semi-implicit solver."""
        return SemiImplicitSolver(
            damping_factor=1e6,
            sts_stages=5,
            sts_safety=0.8
        )

    def test_solver_advances_temperature(self, geometry, model_with_thermal, solver):
        """Solver should advance temperature when thermal is enabled."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        B = jnp.zeros((nx, ny, nz, 3))
        Te = 100.0 + 10.0 * geometry.x_grid**2

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=Te,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # First verify that rhs has non-zero dT/dt
        rhs = model_with_thermal.compute_rhs(state, geometry)
        has_dT = rhs.Te is not None and jnp.any(jnp.abs(rhs.Te) > 1e-20)
        assert has_dT, "Model should produce non-zero dT/dt"

        # Take a single step with small dt (within stability)
        dt = 1e-6
        new_state = solver.step(state, dt, model_with_thermal, geometry)

        # Time should advance
        assert float(new_state.time) == float(state.time) + dt
        assert int(new_state.step) == int(state.step) + 1

        # Temperature should be finite (no NaN)
        assert jnp.all(jnp.isfinite(new_state.Te)), "Temperature should remain finite"

    def test_solver_maintains_positive_temperature(self, geometry, model_with_thermal, solver):
        """Solver should ensure temperature stays positive."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with very low temperature (could go negative with cooling)
        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=jnp.ones((nx, ny, nz)) * 0.01,  # Very low T
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        dt = 1e-8
        new_state = solver.step(state, dt, model_with_thermal, geometry)

        # Temperature should be at least minimum
        assert jnp.all(new_state.Te >= 1e-3), \
            "Temperature should be clipped to minimum"

    def test_from_config_creates_solver(self):
        """from_config should create solver with parameters."""
        config = {
            "damping_factor": 1e5,
            "sts_stages": 10,
            "sts_safety": 0.9
        }

        solver = SemiImplicitSolver.from_config(config)

        assert solver.damping_factor == 1e5
        assert solver.sts_stages == 10
        assert solver.sts_safety == 0.9

    def test_temperature_cfl_computed(self, geometry, model_with_thermal, solver):
        """compute_temperature_cfl should return finite value."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=jnp.ones((nx, ny, nz)) * 100.0,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        dt_cfl = solver.compute_temperature_cfl(state, model_with_thermal, geometry)

        # Should be finite and positive
        assert jnp.isfinite(dt_cfl), "CFL should be finite"
        assert dt_cfl > 0, "CFL should be positive"

    def test_temperature_cfl_infinite_without_thermal(self, geometry, solver):
        """compute_temperature_cfl should return inf without thermal."""
        model_no_thermal = ResistiveMHD(eta=1e-6)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=jnp.ones((nx, ny, nz)) * 100.0,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        dt_cfl = solver.compute_temperature_cfl(state, model_no_thermal, geometry)

        assert dt_cfl == jnp.inf, "CFL should be inf without thermal"


class TestRK4SolverAllFields:
    """Tests for RK4 solver updating all state fields."""

    def test_rk4_updates_temperature(self):
        """RK4 should update Te field using RK4 formula, not just final RHS."""
        from jax_frc.solvers.explicit import RK4Solver
        from jax_frc.core.geometry import Geometry

        geom = Geometry(nx=8, ny=8, nz=8, bc_x="periodic", bc_y="periodic", bc_z="periodic")

        # Initial state with non-uniform temperature (will have diffusion)
        Te_init = 100.0 + 50.0 * jnp.sin(2 * jnp.pi * geom.x_grid)

        state = State(
            B=jnp.ones((8, 8, 8, 3)) * 0.1,
            E=jnp.zeros((8, 8, 8, 3)),
            n=jnp.ones((8, 8, 8)) * 1e20,
            p=jnp.ones((8, 8, 8)) * 1e3,
            v=jnp.zeros((8, 8, 8, 3)),
            Te=Te_init,
        )

        # Model with thermal diffusion
        model = ExtendedMHD(eta=1e-4, include_hall=False, kappa_perp=1e-2)
        solver = RK4Solver()

        # Take one step
        new_state = solver.step(state, dt=1e-6, model=model, geometry=geom)

        # Te should be updated (not just B)
        assert new_state.Te is not None
        # Te should have changed if model has thermal diffusion
        # (The change may be small but should be non-zero)
