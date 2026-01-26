"""Tests for energy partition diagnostics."""

import pytest
import jax.numpy as jnp

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.energy import EnergyDiagnostics
from jax_frc.constants import MU0, MI


class TestEnergyDiagnostics:
    """Tests for EnergyDiagnostics probe."""

    @pytest.fixture
    def geometry(self):
        """Create a standard cylindrical geometry for tests."""
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0
        )

    @pytest.fixture
    def zero_state(self, geometry):
        """Create a state with all zeros."""
        return State.zeros(nr=geometry.nr, nz=geometry.nz)

    @pytest.fixture
    def nonzero_B_state(self, geometry):
        """Create a state with nonzero magnetic field."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Create a uniform B_z field of 1 Tesla
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)  # B_z = 1 T

        return state.replace(B=B)

    @pytest.fixture
    def nonzero_v_state(self, geometry):
        """Create a state with nonzero velocity and density."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Create a uniform v_z velocity of 1000 m/s
        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 2].set(1000.0)  # v_z = 1000 m/s

        # Set density to 1e20 m^-3
        n = jnp.ones((geometry.nr, geometry.nz)) * 1e20

        return state.replace(v=v, n=n)

    @pytest.fixture
    def nonzero_p_state(self, geometry):
        """Create a state with nonzero pressure."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Create uniform pressure of 1000 Pa
        p = jnp.ones((geometry.nr, geometry.nz)) * 1000.0

        return state.replace(p=p)

    # ==========================================================================
    # Basic functionality tests
    # ==========================================================================

    def test_name_is_energy_partition(self):
        """Diagnostic should have correct name."""
        diag = EnergyDiagnostics()
        assert diag.name == "energy_partition"

    def test_compute_returns_all_energy_components(self, zero_state, geometry):
        """compute() should return dict with all energy types."""
        diag = EnergyDiagnostics()
        result = diag.compute(zero_state, geometry)

        assert "E_magnetic" in result
        assert "E_kinetic" in result
        assert "E_thermal" in result
        assert "E_total" in result

    def test_measure_returns_total_energy(self, zero_state, geometry):
        """measure() should return total energy."""
        diag = EnergyDiagnostics()

        total = diag.measure(zero_state, geometry)
        result = diag.compute(zero_state, geometry)

        assert total == result["E_total"]

    # ==========================================================================
    # Zero state tests
    # ==========================================================================

    def test_zero_state_gives_zero_magnetic_energy(self, zero_state, geometry):
        """Zero B field gives zero magnetic energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(zero_state, geometry)

        assert result["E_magnetic"] == 0.0

    def test_zero_state_gives_zero_kinetic_energy(self, zero_state, geometry):
        """Zero velocity gives zero kinetic energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(zero_state, geometry)

        assert result["E_kinetic"] == 0.0

    def test_zero_state_gives_zero_thermal_energy(self, zero_state, geometry):
        """Zero pressure gives zero thermal energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(zero_state, geometry)

        assert result["E_thermal"] == 0.0

    def test_zero_state_gives_zero_total_energy(self, zero_state, geometry):
        """Zero state gives zero total energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(zero_state, geometry)

        assert result["E_total"] == 0.0

    # ==========================================================================
    # Non-negativity tests
    # ==========================================================================

    def test_magnetic_energy_nonnegative(self, nonzero_B_state, geometry):
        """Magnetic energy should always be non-negative."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_B_state, geometry)

        assert result["E_magnetic"] >= 0.0

    def test_kinetic_energy_nonnegative(self, nonzero_v_state, geometry):
        """Kinetic energy should always be non-negative."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_v_state, geometry)

        assert result["E_kinetic"] >= 0.0

    def test_thermal_energy_nonnegative(self, nonzero_p_state, geometry):
        """Thermal energy should always be non-negative."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_p_state, geometry)

        assert result["E_thermal"] >= 0.0

    # ==========================================================================
    # Energy additivity tests
    # ==========================================================================

    def test_total_equals_sum_of_components(self, geometry):
        """E_total = E_magnetic + E_kinetic + E_thermal."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Set up all non-zero fields
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(0.5)  # B_z = 0.5 T

        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 0].set(500.0)  # v_r = 500 m/s

        n = jnp.ones((geometry.nr, geometry.nz)) * 1e19
        p = jnp.ones((geometry.nr, geometry.nz)) * 500.0

        state = state.replace(B=B, v=v, n=n, p=p)

        diag = EnergyDiagnostics()
        result = diag.compute(state, geometry)

        expected_total = result["E_magnetic"] + result["E_kinetic"] + result["E_thermal"]
        assert abs(result["E_total"] - expected_total) < 1e-10

    # ==========================================================================
    # Nonzero field tests
    # ==========================================================================

    def test_nonzero_B_gives_nonzero_magnetic_energy(self, nonzero_B_state, geometry):
        """Nonzero B field gives nonzero magnetic energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_B_state, geometry)

        assert result["E_magnetic"] > 0.0

    def test_nonzero_v_gives_nonzero_kinetic_energy(self, nonzero_v_state, geometry):
        """Nonzero velocity gives nonzero kinetic energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_v_state, geometry)

        assert result["E_kinetic"] > 0.0

    def test_nonzero_p_gives_nonzero_thermal_energy(self, nonzero_p_state, geometry):
        """Nonzero pressure gives nonzero thermal energy."""
        diag = EnergyDiagnostics()
        result = diag.compute(nonzero_p_state, geometry)

        assert result["E_thermal"] > 0.0

    # ==========================================================================
    # Physical scaling tests
    # ==========================================================================

    def test_magnetic_energy_scales_with_B_squared(self, geometry):
        """Magnetic energy should scale as B^2."""
        state1 = State.zeros(nr=geometry.nr, nz=geometry.nz)
        state2 = State.zeros(nr=geometry.nr, nz=geometry.nz)

        B1 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B1 = B1.at[:, :, 2].set(1.0)

        B2 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B2 = B2.at[:, :, 2].set(2.0)  # Double the field

        state1 = state1.replace(B=B1)
        state2 = state2.replace(B=B2)

        diag = EnergyDiagnostics()
        E1 = diag.compute(state1, geometry)["E_magnetic"]
        E2 = diag.compute(state2, geometry)["E_magnetic"]

        # E should scale as B^2, so doubling B should quadruple E
        assert abs(E2 / E1 - 4.0) < 1e-10

    def test_kinetic_energy_scales_with_v_squared(self, geometry):
        """Kinetic energy should scale as v^2."""
        state1 = State.zeros(nr=geometry.nr, nz=geometry.nz)
        state2 = State.zeros(nr=geometry.nr, nz=geometry.nz)

        n = jnp.ones((geometry.nr, geometry.nz)) * 1e20

        v1 = jnp.zeros((geometry.nr, geometry.nz, 3))
        v1 = v1.at[:, :, 2].set(1000.0)

        v2 = jnp.zeros((geometry.nr, geometry.nz, 3))
        v2 = v2.at[:, :, 2].set(2000.0)  # Double the velocity

        state1 = state1.replace(v=v1, n=n)
        state2 = state2.replace(v=v2, n=n)

        diag = EnergyDiagnostics()
        E1 = diag.compute(state1, geometry)["E_kinetic"]
        E2 = diag.compute(state2, geometry)["E_kinetic"]

        # E should scale as v^2, so doubling v should quadruple E
        assert abs(E2 / E1 - 4.0) < 1e-10

    def test_thermal_energy_scales_with_pressure(self, geometry):
        """Thermal energy should scale linearly with pressure."""
        state1 = State.zeros(nr=geometry.nr, nz=geometry.nz)
        state2 = State.zeros(nr=geometry.nr, nz=geometry.nz)

        p1 = jnp.ones((geometry.nr, geometry.nz)) * 1000.0
        p2 = jnp.ones((geometry.nr, geometry.nz)) * 2000.0  # Double pressure

        state1 = state1.replace(p=p1)
        state2 = state2.replace(p=p2)

        diag = EnergyDiagnostics()
        E1 = diag.compute(state1, geometry)["E_thermal"]
        E2 = diag.compute(state2, geometry)["E_thermal"]

        # E should scale linearly with p
        assert abs(E2 / E1 - 2.0) < 1e-10

    def test_kinetic_energy_scales_with_density(self, geometry):
        """Kinetic energy should scale linearly with density."""
        state1 = State.zeros(nr=geometry.nr, nz=geometry.nz)
        state2 = State.zeros(nr=geometry.nr, nz=geometry.nz)

        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 2].set(1000.0)

        n1 = jnp.ones((geometry.nr, geometry.nz)) * 1e20
        n2 = jnp.ones((geometry.nr, geometry.nz)) * 2e20  # Double density

        state1 = state1.replace(v=v, n=n1)
        state2 = state2.replace(v=v, n=n2)

        diag = EnergyDiagnostics()
        E1 = diag.compute(state1, geometry)["E_kinetic"]
        E2 = diag.compute(state2, geometry)["E_kinetic"]

        # E should scale linearly with density
        assert abs(E2 / E1 - 2.0) < 1e-10

    # ==========================================================================
    # Custom parameter tests
    # ==========================================================================

    def test_custom_gamma(self, nonzero_p_state, geometry):
        """Custom gamma affects thermal energy calculation."""
        diag_default = EnergyDiagnostics()  # gamma = 5/3
        diag_custom = EnergyDiagnostics(gamma=1.4)

        E_default = diag_default.compute(nonzero_p_state, geometry)["E_thermal"]
        E_custom = diag_custom.compute(nonzero_p_state, geometry)["E_thermal"]

        # E_thermal = p / (gamma - 1), so E_custom / E_default = (gamma_default - 1) / (gamma_custom - 1)
        expected_ratio = (5.0 / 3.0 - 1.0) / (1.4 - 1.0)
        actual_ratio = E_custom / E_default

        # Allow small tolerance for floating point differences
        assert abs(actual_ratio - expected_ratio) < 1e-5

    def test_custom_ion_mass(self, nonzero_v_state, geometry):
        """Custom ion mass affects kinetic energy calculation."""
        diag_default = EnergyDiagnostics()  # ion_mass = MI
        diag_custom = EnergyDiagnostics(ion_mass=2.0 * MI)  # Double mass

        E_default = diag_default.compute(nonzero_v_state, geometry)["E_kinetic"]
        E_custom = diag_custom.compute(nonzero_v_state, geometry)["E_kinetic"]

        # E_kinetic scales linearly with mass
        assert abs(E_custom / E_default - 2.0) < 1e-10

    # ==========================================================================
    # Analytical verification tests
    # ==========================================================================

    def test_magnetic_energy_analytical(self):
        """Test magnetic energy against analytical calculation."""
        # Use simple geometry for analytical calculation
        # Higher resolution for better accuracy
        geometry = Geometry(
            coord_system="cylindrical",
            nr=200, nz=200,
            r_min=0.5, r_max=1.5,  # 1m width
            z_min=-0.5, z_max=0.5  # 1m height
        )

        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Uniform B_z = 1 T
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)
        state = state.replace(B=B)

        diag = EnergyDiagnostics()
        E_computed = diag.compute(state, geometry)["E_magnetic"]

        # Analytical: E = (1/2mu0) * B^2 * V
        # V = pi * (r_max^2 - r_min^2) * L_z for cylindrical
        # V = pi * (1.5^2 - 0.5^2) * 1.0 = pi * 2.0
        volume = jnp.pi * (geometry.r_max**2 - geometry.r_min**2) * (geometry.z_max - geometry.z_min)
        B_val = 1.0
        E_analytical = (1.0 / (2.0 * MU0)) * B_val**2 * volume

        # Allow 3% error due to discretization
        relative_error = abs(E_computed - E_analytical) / E_analytical
        assert relative_error < 0.03

    def test_thermal_energy_analytical(self):
        """Test thermal energy against analytical calculation."""
        # Use simple geometry for analytical calculation
        # Higher resolution for better accuracy
        geometry = Geometry(
            coord_system="cylindrical",
            nr=200, nz=200,
            r_min=0.5, r_max=1.5,
            z_min=-0.5, z_max=0.5
        )

        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Uniform pressure = 1000 Pa
        p = jnp.ones((geometry.nr, geometry.nz)) * 1000.0
        state = state.replace(p=p)

        gamma = 5.0 / 3.0
        diag = EnergyDiagnostics(gamma=gamma)
        E_computed = diag.compute(state, geometry)["E_thermal"]

        # Analytical: E = p / (gamma - 1) * V
        volume = jnp.pi * (geometry.r_max**2 - geometry.r_min**2) * (geometry.z_max - geometry.z_min)
        p_val = 1000.0
        E_analytical = p_val / (gamma - 1.0) * volume

        # Allow 3% error due to discretization
        relative_error = abs(E_computed - E_analytical) / E_analytical
        assert relative_error < 0.03

    # ==========================================================================
    # All B components test
    # ==========================================================================

    def test_magnetic_energy_includes_all_components(self, geometry):
        """Magnetic energy includes B_r, B_phi, and B_z."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Set only B_r
        B_r = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_r = B_r.at[:, :, 0].set(1.0)

        # Set only B_phi
        B_phi = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_phi = B_phi.at[:, :, 1].set(1.0)

        # Set only B_z
        B_z = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_z = B_z.at[:, :, 2].set(1.0)

        diag = EnergyDiagnostics()

        E_r = diag.compute(state.replace(B=B_r), geometry)["E_magnetic"]
        E_phi = diag.compute(state.replace(B=B_phi), geometry)["E_magnetic"]
        E_z = diag.compute(state.replace(B=B_z), geometry)["E_magnetic"]

        # All should be equal for same magnitude
        assert abs(E_r - E_phi) < 1e-10
        assert abs(E_r - E_z) < 1e-10
        assert E_r > 0.0

    # ==========================================================================
    # All velocity components test
    # ==========================================================================

    def test_kinetic_energy_includes_all_components(self, geometry):
        """Kinetic energy includes v_r, v_phi, and v_z."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)
        n = jnp.ones((geometry.nr, geometry.nz)) * 1e20

        # Set only v_r
        v_r = jnp.zeros((geometry.nr, geometry.nz, 3))
        v_r = v_r.at[:, :, 0].set(1000.0)

        # Set only v_phi
        v_phi = jnp.zeros((geometry.nr, geometry.nz, 3))
        v_phi = v_phi.at[:, :, 1].set(1000.0)

        # Set only v_z
        v_z = jnp.zeros((geometry.nr, geometry.nz, 3))
        v_z = v_z.at[:, :, 2].set(1000.0)

        diag = EnergyDiagnostics()

        E_r = diag.compute(state.replace(v=v_r, n=n), geometry)["E_kinetic"]
        E_phi = diag.compute(state.replace(v=v_phi, n=n), geometry)["E_kinetic"]
        E_z = diag.compute(state.replace(v=v_z, n=n), geometry)["E_kinetic"]

        # All should be equal for same magnitude
        assert abs(E_r - E_phi) < 1e-10
        assert abs(E_r - E_z) < 1e-10
        assert E_r > 0.0

    # ==========================================================================
    # Zero velocity but nonzero density test
    # ==========================================================================

    def test_zero_velocity_nonzero_density_gives_zero_kinetic(self, geometry):
        """Zero velocity with nonzero density gives zero kinetic energy."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Nonzero density but zero velocity
        n = jnp.ones((geometry.nr, geometry.nz)) * 1e20
        state = state.replace(n=n)

        diag = EnergyDiagnostics()
        result = diag.compute(state, geometry)

        assert result["E_kinetic"] == 0.0

    def test_nonzero_velocity_zero_density_gives_zero_kinetic(self, geometry):
        """Nonzero velocity with zero density gives zero kinetic energy."""
        state = State.zeros(nr=geometry.nr, nz=geometry.nz)

        # Nonzero velocity but zero density (default)
        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 2].set(1000.0)
        state = state.replace(v=v)

        diag = EnergyDiagnostics()
        result = diag.compute(state, geometry)

        assert result["E_kinetic"] == 0.0
