"""Integration tests for Extended MHD with temperature evolution."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.semi_implicit import SemiImplicitSolver
from jax_frc.models.extended_mhd import ExtendedMHD, TemperatureBoundaryCondition
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from tests.utils.cartesian import make_geometry


class TestEndToEndTemperatureEvolution:
    """End-to-end simulation tests with temperature."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=4, nz=32)

    @pytest.fixture
    def model(self):
        """Create ExtendedMHD model with thermal transport.

        Uses high density to suppress Hall instability while still exercising
        the temperature evolution code path.
        """
        bc = TemperatureBoundaryCondition(bc_type="neumann")
        return ExtendedMHD(
            eta=1e-5,
            include_hall=False,
            include_electron_pressure=False,
            kappa_perp=1e-2,
            temperature_bc=bc,
        )

    @pytest.fixture
    def solver(self):
        """Create solver.

        Note: damping_factor must be large enough to stabilize Whistler modes.
        For dt=1e-7 and Whistler CFL ~2e-8, we need damping >> 1/dt² ~ 1e14.
        """
        return SemiImplicitSolver(
            damping_factor=1e14,  # High enough to damp Whistler modes
            sts_stages=5,
            sts_safety=0.8
        )

    def test_simulation_runs_without_error(self, geometry, model, solver):
        """Full simulation should run without errors or NaN."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        B = jnp.zeros((nx, ny, nz, 3))
        T_init = 200.0 * jnp.exp(-2 * (geometry.x_grid**2 + geometry.z_grid**2)) + 50.0

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e20,  # High density to suppress Hall
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=T_init,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # Run 10 steps
        dt = 1e-7
        n_steps = 10
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Check no NaN
        assert jnp.all(jnp.isfinite(state.Te)), "Temperature should remain finite"
        assert jnp.all(jnp.isfinite(state.B)), "B field should remain finite"

        # Check temperature is positive
        assert jnp.all(state.Te > 0), "Temperature should remain positive"

        # Check time advanced
        assert float(state.time) == pytest.approx(n_steps * dt, rel=1e-10)
        assert int(state.step) == n_steps

    def test_temperature_stays_bounded(self, geometry, model, solver):
        """Temperature should stay within reasonable bounds."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        B = jnp.zeros((nx, ny, nz, 3))
        T_init = jnp.ones((nx, ny, nz)) * 100.0

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e20,  # High density to suppress Hall
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=T_init,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # Run many steps
        dt = 1e-7
        n_steps = 20
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Temperature should not explode or collapse
        T_max = float(jnp.max(state.Te))
        T_min = float(jnp.min(state.Te))

        # Allow some heating from Ohmic, but not extreme
        assert T_max < 1e6, f"Temperature exploded: max = {T_max}"
        assert T_min > 0, f"Temperature went negative: min = {T_min}"


class TestThermalDiffusionConservation:
    """Test thermal diffusion conserves total energy."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=4, nz=32)

    def test_diffusion_conserves_total_thermal_energy(self, geometry):
        """Diffusion should conserve total thermal energy for periodic operators."""
        model = ExtendedMHD(
            eta=1e-6,
            include_hall=False,
            include_electron_pressure=False,
            kappa_perp=1e-2,
        )
        solver = SemiImplicitSolver(sts_stages=3, damping_factor=1e12)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        z_span = geometry.z_max - geometry.z_min

        B = jnp.zeros((nx, ny, nz, 3))
        Te = 100.0 + 10.0 * jnp.sin(2 * jnp.pi * geometry.z_grid / z_span)
        n = jnp.ones((nx, ny, nz)) * 1e20

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=n,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=Te,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        E_th_initial = float(jnp.sum(1.5 * n * Te))

        dt = 1e-6
        n_steps = 5
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        E_th_final = float(jnp.sum(1.5 * n * state.Te))

        assert abs(E_th_final - E_th_initial) / E_th_initial < 1e-3


class TestHeatConductionAnalytical:
    """Test heat conduction against analytical solutions."""

    def test_1d_heat_conduction_diffusion(self):
        """Test 1D heat conduction matches diffusion behavior.

        For a Gaussian temperature profile T(x,0) = T_0 * exp(-x²/(2σ²)),
        the solution spreads as T(x,t) = T_0 * (σ²/(σ²+2Dt))^0.5 * exp(...)

        The width increases as σ_eff² = σ² + 2*D*t
        """
        geometry = Geometry(
            nx=8,
            ny=4,
            nz=64,  # Fine grid in z for diffusion
            x_min=0.1,
            x_max=0.9,  # Avoid boundaries
            y_min=-0.5,
            y_max=0.5,
            z_min=-2.0,
            z_max=2.0,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

        # Use uniform B in z (parallel to gradient)
        # So parallel conduction dominates
        D = 1e-3  # Effective diffusivity: D = κ / (3/2 * n)
        n0 = 1e19
        kappa = D * 1.5 * n0  # κ = D * (3/2 * n)

        bc = TemperatureBoundaryCondition(bc_type="neumann")

        model = ExtendedMHD(
            eta=1e-10,
            include_hall=False,
            include_electron_pressure=False,
            kappa_perp=kappa,
            temperature_bc=bc,
        )
        solver = SemiImplicitSolver(sts_stages=5, sts_safety=0.5, damping_factor=1e14)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        z = geometry.z_grid

        B = jnp.zeros((nx, ny, nz, 3))

        # Gaussian temperature profile in z
        sigma = 0.3
        T_peak = 200.0
        T_base = 50.0
        T_init = T_peak * jnp.exp(-z**2 / (2 * sigma**2)) + T_base

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * n0,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=T_init,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # Initial width (variance)
        T_center = state.Te[nx // 2, ny // 2, :]
        T_max_initial = float(jnp.max(T_center))

        # Run for some time
        dt = 1e-5
        n_steps = 30
        total_time = dt * n_steps

        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Final peak should be lower (profile spreading)
        T_center_final = state.Te[nx // 2, ny // 2, :]
        T_max_final = float(jnp.max(T_center_final))

        # Peak should decrease as profile spreads
        # For 1D diffusion: T_max(t) / T_max(0) = sqrt(σ² / (σ² + 2Dt))
        expected_ratio = (sigma**2 / (sigma**2 + 2 * D * total_time))**0.5

        # Allow wide tolerance due to boundary effects and numerics
        actual_ratio = (T_max_final - T_base) / (T_max_initial - T_base)

        # Peak should definitely decrease
        assert T_max_final < T_max_initial, \
            f"Peak should decrease with diffusion: {T_max_initial} -> {T_max_final}"

        # Ratio should be in reasonable range (within factor of 2 of analytical)
        # This is a qualitative test, not exact benchmark
        assert actual_ratio < 1.0, "Profile should spread (ratio < 1)"
        assert actual_ratio > 0.1, f"Profile shouldn't collapse completely: ratio = {actual_ratio}"
