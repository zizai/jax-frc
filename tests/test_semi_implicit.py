"""Tests for Semi-Implicit Solver with temperature evolution."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.semi_implicit import SemiImplicitSolver
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel, TemperatureBoundaryCondition
from jax_frc.models.energy import ThermalTransport
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestSemiImplicitSolverTemperature:
    """Tests for temperature handling in SemiImplicitSolver."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

    @pytest.fixture
    def model_with_thermal(self):
        """Create ExtendedMHD model with thermal transport."""
        resistivity = SpitzerResistivity(eta_0=1e-4)
        halo = HaloDensityModel()
        thermal = ThermalTransport(
            kappa_parallel_0=1e10,
            kappa_perp_ratio=1e-6,
            use_spitzer=False
        )
        bc = TemperatureBoundaryCondition(bc_type="neumann")
        return ExtendedMHD(
            resistivity=resistivity,
            halo_model=halo,
            thermal=thermal,
            temperature_bc=bc
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
        nr, nz = geometry.nr, geometry.nz

        # Create state with B field that produces current
        r = geometry.r_grid
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.1 * jnp.exp(-r**2))

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # First verify that rhs has non-zero dT/dt
        rhs = model_with_thermal.compute_rhs(state, geometry)
        has_dT = jnp.any(jnp.abs(rhs.T) > 1e-20)
        assert has_dT, "Model should produce non-zero dT/dt"

        # Take a single step with small dt (within stability)
        dt = 1e-6
        new_state = solver.step(state, dt, model_with_thermal, geometry)

        # Time should advance
        assert new_state.time == state.time + dt
        assert new_state.step == state.step + 1

        # Temperature should be finite (no NaN)
        assert jnp.all(jnp.isfinite(new_state.T)), "Temperature should remain finite"

    def test_solver_maintains_positive_temperature(self, geometry, model_with_thermal, solver):
        """Solver should ensure temperature stays positive."""
        nr, nz = geometry.nr, geometry.nz

        # Create state with very low temperature (could go negative with cooling)
        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 0.01,  # Very low T
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        dt = 1e-8
        new_state = solver.step(state, dt, model_with_thermal, geometry)

        # Temperature should be at least minimum
        assert jnp.all(new_state.T >= 1e-3), \
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
        nr, nz = geometry.nr, geometry.nz

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        dt_cfl = solver.compute_temperature_cfl(state, model_with_thermal, geometry)

        # Should be finite and positive
        assert jnp.isfinite(dt_cfl), "CFL should be finite"
        assert dt_cfl > 0, "CFL should be positive"

    def test_temperature_cfl_infinite_without_thermal(self, geometry, solver):
        """compute_temperature_cfl should return inf without thermal."""
        resistivity = SpitzerResistivity(eta_0=1e-6)
        halo = HaloDensityModel()
        model_no_thermal = ExtendedMHD(
            resistivity=resistivity,
            halo_model=halo,
            thermal=None
        )

        nr, nz = geometry.nr, geometry.nz
        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        dt_cfl = solver.compute_temperature_cfl(state, model_no_thermal, geometry)

        assert dt_cfl == jnp.inf, "CFL should be inf without thermal"
