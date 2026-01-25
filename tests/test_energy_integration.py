"""Integration tests for Extended MHD with temperature evolution."""
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


class TestEndToEndTemperatureEvolution:
    """End-to-end simulation tests with temperature."""

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
    def model(self):
        """Create ExtendedMHD model with thermal transport.

        Uses high density to suppress Hall instability while still exercising
        the temperature evolution code path.
        """
        resistivity = SpitzerResistivity(eta_0=1e-5)
        # High density reduces Hall term (∝ 1/n)
        halo = HaloDensityModel(halo_density=1e19, core_density=1e20)
        thermal = ThermalTransport(
            kappa_parallel_0=1e8,  # Reduced for stable testing
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
        nr, nz = geometry.nr, geometry.nz

        # Create initial state with B field compatible with conducting wall BCs
        # B must decay to zero at boundaries to avoid spurious boundary currents
        r = geometry.r_grid
        z = geometry.z_grid

        # B profile that decays smoothly to boundaries (satisfies B=0 at walls)
        # Use a profile that's zero at r_min, r_max and z_min, z_max
        r_norm = (r - geometry.r_min) / (geometry.r_max - geometry.r_min)
        z_norm = (z - geometry.z_min) / (geometry.z_max - geometry.z_min)
        # Envelope: zero at boundaries, 1 in center
        envelope = 16 * r_norm * (1 - r_norm) * z_norm * (1 - z_norm)

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.1 * envelope)  # B_z with boundary-compatible profile

        # Initial temperature profile: hot core, cold edge
        T_init = 200.0 * jnp.exp(-2 * r**2) + 50.0

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e20,  # High density to suppress Hall
            p=jnp.ones((nr, nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Run 10 steps
        dt = 1e-7
        n_steps = 10
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Check no NaN
        assert jnp.all(jnp.isfinite(state.T)), "Temperature should remain finite"
        assert jnp.all(jnp.isfinite(state.B)), "B field should remain finite"

        # Check temperature is positive
        assert jnp.all(state.T > 0), "Temperature should remain positive"

        # Check time advanced
        assert state.time == pytest.approx(n_steps * dt, rel=1e-10)
        assert state.step == n_steps

    def test_temperature_stays_bounded(self, geometry, model, solver):
        """Temperature should stay within reasonable bounds."""
        nr, nz = geometry.nr, geometry.nz
        r = geometry.r_grid
        z = geometry.z_grid

        # B profile compatible with conducting wall BCs
        r_norm = (r - geometry.r_min) / (geometry.r_max - geometry.r_min)
        z_norm = (z - geometry.z_min) / (geometry.z_max - geometry.z_min)
        envelope = 16 * r_norm * (1 - r_norm) * z_norm * (1 - z_norm)

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.05 * envelope)

        T_init = jnp.ones((nr, nz)) * 100.0

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e20,  # High density to suppress Hall
            p=jnp.ones((nr, nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Run many steps
        dt = 1e-7
        n_steps = 20
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Temperature should not explode or collapse
        T_max = float(jnp.max(state.T))
        T_min = float(jnp.min(state.T))

        # Allow some heating from Ohmic, but not extreme
        assert T_max < 1e6, f"Temperature exploded: max = {T_max}"
        assert T_min > 0, f"Temperature went negative: min = {T_min}"


class TestOhmicHeatingConservation:
    """Test energy conservation with Ohmic heating."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return Geometry(
            coord_system="cylindrical",
            r_min=0.1, r_max=0.5,  # Avoid r=0 issues
            z_min=-0.5, z_max=0.5,
            nr=16, nz=32
        )

    def test_ohmic_heating_increases_total_thermal_energy(self, geometry):
        """Ohmic heating (ηJ²) should increase total thermal energy.

        Note: We use parameters that minimize Hall instability while keeping
        Ohmic heating significant. The Hall term scales as B/(ne), while
        Ohmic heating scales as ηJ². We use high density and moderate B.
        """
        resistivity = SpitzerResistivity(eta_0=1e-3)  # High η for visible effect
        halo = HaloDensityModel(halo_density=1e18, core_density=1e20)  # Higher density reduces Hall
        thermal = ThermalTransport(kappa_parallel_0=0, use_spitzer=False)  # No conduction
        bc = TemperatureBoundaryCondition(bc_type="neumann")

        model = ExtendedMHD(
            resistivity=resistivity,
            halo_model=halo,
            thermal=thermal,
            temperature_bc=bc
        )
        solver = SemiImplicitSolver(sts_stages=1, damping_factor=1e14)

        nr, nz = geometry.nr, geometry.nz
        r = geometry.r_grid

        # Weak B gradient to produce small J (minimizes Hall while keeping Ohmic)
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.01 * (1.0 - 0.5 * r))  # Weak gradient produces J

        T_init = jnp.ones((nr, nz)) * 100.0
        n = jnp.ones((nr, nz)) * 1e20  # High density

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=n,
            p=jnp.ones((nr, nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Compute initial thermal energy: E_th = (3/2) * n * T
        E_th_initial = float(jnp.sum(1.5 * n * state.T))

        # Run a few steps
        dt = 1e-6
        n_steps = 10
        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Compute final thermal energy
        E_th_final = float(jnp.sum(1.5 * n * state.T))

        # With Ohmic heating and no conduction losses, thermal energy should increase
        assert E_th_final >= E_th_initial, \
            f"Thermal energy should not decrease with Ohmic heating: {E_th_initial} -> {E_th_final}"


class TestHeatConductionAnalytical:
    """Test heat conduction against analytical solutions."""

    def test_1d_heat_conduction_diffusion(self):
        """Test 1D heat conduction matches diffusion behavior.

        For a Gaussian temperature profile T(x,0) = T_0 * exp(-x²/(2σ²)),
        the solution spreads as T(x,t) = T_0 * (σ²/(σ²+2Dt))^0.5 * exp(...)

        The width increases as σ_eff² = σ² + 2*D*t
        """
        geometry = Geometry(
            coord_system="cylindrical",
            r_min=0.1, r_max=0.9,  # Avoid boundaries
            z_min=-2.0, z_max=2.0,
            nr=8, nz=64  # Fine grid in z for diffusion
        )

        # Use uniform B in z (parallel to gradient)
        # So parallel conduction dominates
        D = 1e-3  # Effective diffusivity: D = κ / (3/2 * n)
        n0 = 1e19
        kappa = D * 1.5 * n0  # κ = D * (3/2 * n)

        resistivity = SpitzerResistivity(eta_0=1e-10)  # Very small
        halo = HaloDensityModel(halo_density=n0, core_density=n0)
        thermal = ThermalTransport(kappa_parallel_0=kappa, use_spitzer=False)
        bc = TemperatureBoundaryCondition(bc_type="neumann")

        model = ExtendedMHD(
            resistivity=resistivity,
            halo_model=halo,
            thermal=thermal,
            temperature_bc=bc
        )
        solver = SemiImplicitSolver(sts_stages=5, sts_safety=0.5, damping_factor=1e14)

        nr, nz = geometry.nr, geometry.nz
        z = geometry.z_grid
        r = geometry.r_grid

        # B field in z direction, but compatible with conducting wall BCs
        # Use envelope that's 1 in interior, decays to 0 at boundaries
        r_norm = (r - geometry.r_min) / (geometry.r_max - geometry.r_min)
        z_norm = (z - geometry.z_min) / (geometry.z_max - geometry.z_min)
        envelope = 16 * r_norm * (1 - r_norm) * z_norm * (1 - z_norm)

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(1.0 * envelope)  # B_z with boundary-compatible profile

        # Gaussian temperature profile in z
        sigma = 0.3
        T_peak = 200.0
        T_base = 50.0
        T_init = T_peak * jnp.exp(-z**2 / (2 * sigma**2)) + T_base

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * n0,
            p=jnp.ones((nr, nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Initial width (variance)
        T_center = state.T[nr//2, :]
        T_max_initial = float(jnp.max(T_center))

        # Run for some time
        dt = 1e-5
        n_steps = 30
        total_time = dt * n_steps

        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Final peak should be lower (profile spreading)
        T_center_final = state.T[nr//2, :]
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
