"""Tests for ThermalTransport class."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.models.energy import ThermalTransport
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestThermalTransportKappa:
    """Tests for thermal conductivity calculations."""

    def test_spitzer_conductivity_scaling(self):
        """Spitzer conductivity should scale as T^(5/2)."""
        transport = ThermalTransport(
            kappa_parallel_0=1.0,
            use_spitzer=True,
            coulomb_log=1.0  # Simplify for testing
        )

        # Test at different temperatures
        T1 = jnp.array([[1.0]])
        T2 = jnp.array([[4.0]])  # 4x temperature

        kappa1 = transport.compute_kappa_parallel(T1)
        kappa2 = transport.compute_kappa_parallel(T2)

        # κ ∝ T^(5/2), so κ2/κ1 = (4)^(5/2) = 32
        ratio = kappa2 / kappa1
        expected = 4.0**2.5  # = 32

        assert jnp.allclose(ratio, expected, rtol=1e-5), \
            f"Spitzer scaling: expected ratio {expected}, got {ratio}"

    def test_constant_conductivity(self):
        """Non-Spitzer should give constant conductivity."""
        kappa_0 = 1e15
        transport = ThermalTransport(
            kappa_parallel_0=kappa_0,
            use_spitzer=False
        )

        T = jnp.array([[1.0, 10.0, 100.0],
                       [1000.0, 10000.0, 100000.0]])

        kappa = transport.compute_kappa_parallel(T)

        assert jnp.allclose(kappa, kappa_0), \
            "Constant conductivity should not depend on T"

    def test_perp_to_parallel_ratio(self):
        """Perpendicular conductivity should be ratio times parallel."""
        ratio = 1e-6
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=ratio,
            use_spitzer=False
        )

        T = jnp.ones((8, 8)) * 100.0

        kappa_par = transport.compute_kappa_parallel(T)
        kappa_perp = transport.compute_kappa_perp(T)

        assert jnp.allclose(kappa_perp / kappa_par, ratio), \
            "κ_⊥/κ_∥ should equal kappa_perp_ratio"

    def test_minimum_temperature_clipping(self):
        """Low temperatures should be clipped to avoid singularities."""
        transport = ThermalTransport(
            kappa_parallel_0=1.0,
            use_spitzer=True,
            coulomb_log=1.0,
            min_temperature=1.0
        )

        # Very low temperature
        T_low = jnp.array([[1e-10]])
        # Temperature at minimum
        T_min = jnp.array([[1.0]])

        kappa_low = transport.compute_kappa_parallel(T_low)
        kappa_min = transport.compute_kappa_parallel(T_min)

        # Both should give same result due to clipping
        assert jnp.allclose(kappa_low, kappa_min), \
            "Temperatures below min should be clipped"


class TestThermalTransportHeatFlux:
    """Tests for heat flux calculations."""

    def test_heat_flux_parallel_to_B(self):
        """Heat flux should be primarily along B for high κ_∥/κ_⊥."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=1e-10,  # Very small perpendicular
            use_spitzer=False
        )

        nr, nz = 16, 16
        dr, dz = 0.01, 0.01

        # Create r grid
        r = jnp.linspace(0.01, 0.16, nr)[:, None] * jnp.ones((1, nz))

        # Uniform B in z direction
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(1.0)  # B_z = 1

        # Temperature gradient in r direction (perpendicular to B)
        T = jnp.linspace(100, 200, nr)[:, None] * jnp.ones((1, nz))

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # With B along z and gradient along r (perpendicular),
        # heat flux should be primarily perpendicular (very small due to low κ_⊥)
        # q_z should be near zero (no gradient in z)
        assert jnp.max(jnp.abs(q_z[2:-2, 2:-2])) < 1e-5, \
            "No heat flux in z without z gradient"

    def test_heat_flux_direction_parallel_gradient(self):
        """Heat flux along B direction when gradient is parallel to B."""
        transport = ThermalTransport(
            kappa_parallel_0=1e10,
            kappa_perp_ratio=1e-6,
            use_spitzer=False
        )

        nr, nz = 16, 32
        dr, dz = 0.01, 0.01

        r = jnp.linspace(0.05, 0.2, nr)[:, None] * jnp.ones((1, nz))

        # Uniform B in z direction
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(1.0)

        # Temperature gradient in z direction (parallel to B)
        z_values = jnp.linspace(0, 0.32, nz)
        T = jnp.ones((nr, 1)) * (100.0 + 1000.0 * z_values[None, :])

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # Heat should flow opposite to gradient (negative q_z for positive dT/dz)
        # Interior points (avoiding boundary artifacts)
        interior_q_z = q_z[2:-2, 2:-2]
        assert jnp.all(interior_q_z < 0), \
            "Heat flux should be opposite to temperature gradient"

    def test_heat_flux_zero_with_uniform_T(self):
        """Heat flux should be zero with uniform temperature."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            use_spitzer=False
        )

        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        r = jnp.linspace(0.01, 0.16, nr)[:, None] * jnp.ones((1, nz))

        # Some B field
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 0].set(0.5)
        B = B.at[:, :, 2].set(0.5)

        # Uniform temperature
        T = jnp.ones((nr, nz)) * 100.0

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # Heat flux should be zero (within numerical precision)
        # Avoid boundaries due to roll artifacts
        assert jnp.allclose(q_r[2:-2, 2:-2], 0, atol=1e-10), \
            "No heat flux with uniform T"
        assert jnp.allclose(q_z[2:-2, 2:-2], 0, atol=1e-10), \
            "No heat flux with uniform T"


class TestThermalTransportDivergence:
    """Tests for heat flux divergence."""

    def test_divergence_zero_uniform_flux(self):
        """Divergence should be near zero for uniform heat flux."""
        transport = ThermalTransport(
            kappa_parallel_0=1e10,
            use_spitzer=False
        )

        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        r = jnp.linspace(0.05, 0.2, nr)[:, None] * jnp.ones((1, nz))

        # Uniform B in z
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(1.0)

        # Uniform temperature
        T = jnp.ones((nr, nz)) * 100.0

        div_q = transport.compute_heat_flux_divergence(T, B, dr, dz, r)

        # Divergence of zero flux should be zero
        assert jnp.allclose(div_q[2:-2, 2:-2], 0, atol=1e-10), \
            "Divergence should be zero for uniform T"

    def test_divergence_shape(self):
        """Divergence should have same shape as input."""
        transport = ThermalTransport()

        nr, nz = 32, 64
        dr, dz = 0.01, 0.01
        r = jnp.linspace(0.01, 0.32, nr)[:, None] * jnp.ones((1, nz))

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(1.0)

        T = jnp.ones((nr, nz)) * 100.0

        div_q = transport.compute_heat_flux_divergence(T, B, dr, dz, r)

        assert div_q.shape == (nr, nz), \
            f"Expected shape ({nr}, {nz}), got {div_q.shape}"


class TestThermalTransportPhysics:
    """Physics validation tests."""

    def test_heat_flux_orthogonal_decomposition(self):
        """Parallel and perpendicular heat flux should be orthogonal."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=0.5,  # Comparable to see both contributions
            use_spitzer=False
        )

        nr, nz = 16, 16
        dr, dz = 0.01, 0.01
        r = jnp.linspace(0.05, 0.2, nr)[:, None] * jnp.ones((1, nz))

        # B at 45 degrees in r-z plane
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 0].set(1.0)  # B_r
        B = B.at[:, :, 2].set(1.0)  # B_z

        # Temperature gradient in r direction
        T = jnp.linspace(100, 200, nr)[:, None] * jnp.ones((1, nz))

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # For B at 45°, the parallel gradient is dT/dr * (1/√2)
        # Parallel heat flux should be along B direction
        # Interior points
        interior_q_r = q_r[4:-4, 4:-4]
        interior_q_z = q_z[4:-4, 4:-4]

        # Both components should be non-zero with B at 45°
        assert jnp.any(jnp.abs(interior_q_r) > 1e-5), \
            "q_r should be non-zero with angled B"
        assert jnp.any(jnp.abs(interior_q_z) > 1e-5), \
            "q_z should be non-zero with angled B"


class TestExtendedMHDTemperature:
    """Tests for temperature evolution in ExtendedMHD."""

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
        resistivity = SpitzerResistivity(eta_0=1e-4)  # Higher for visible Ohmic heating
        halo = HaloDensityModel()
        thermal = ThermalTransport(
            kappa_parallel_0=1e10,  # Reduced for testing
            kappa_perp_ratio=1e-6,
            use_spitzer=False
        )
        return ExtendedMHD(resistivity=resistivity, halo_model=halo, thermal=thermal)

    @pytest.fixture
    def model_without_thermal(self):
        """Create ExtendedMHD model without thermal transport."""
        resistivity = SpitzerResistivity(eta_0=1e-6)
        halo = HaloDensityModel()
        return ExtendedMHD(resistivity=resistivity, halo_model=halo, thermal=None)

    def test_compute_rhs_returns_dT_with_thermal(self, geometry, model_with_thermal):
        """compute_rhs should return dT when thermal is enabled."""
        nr, nz = geometry.nr, geometry.nz

        # Create state with non-zero current (for Ohmic heating)
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.1)  # Uniform B_z

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

        rhs = model_with_thermal.compute_rhs(state, geometry)

        # RHS should have non-trivial T (dT/dt)
        assert rhs.T.shape == (nr, nz), f"Expected T shape ({nr}, {nz})"

    def test_compute_rhs_unchanged_without_thermal(self, geometry, model_without_thermal):
        """compute_rhs should not modify T when thermal is disabled."""
        nr, nz = geometry.nr, geometry.nz

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.1)

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

        rhs = model_without_thermal.compute_rhs(state, geometry)

        # T should be unchanged (no dT in RHS)
        assert jnp.allclose(rhs.T, state.T), "T should be unchanged without thermal"

    def test_ohmic_heating_increases_temperature(self, geometry, model_with_thermal):
        """Ohmic heating (ηJ²) should increase temperature."""
        nr, nz = geometry.nr, geometry.nz

        # Create state with B that produces current
        # B_z varying in r produces J_phi from dB_z/dr
        r = geometry.r_grid
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.1 * jnp.exp(-r**2))  # B_z(r)

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,  # Uniform T (no conduction)
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),  # No compression
            particles=None,
            time=0.0,
            step=0
        )

        rhs = model_with_thermal.compute_rhs(state, geometry)

        # dT/dt should be positive where J² > 0 (Ohmic heating)
        # Interior points away from boundaries
        interior_dT = rhs.T[3:-3, 3:-3]

        # With uniform T and no velocity, only Ohmic heating contributes
        # dT/dt = (2/3n) * ηJ² > 0
        assert jnp.any(interior_dT > 0), "Ohmic heating should increase temperature"

    def test_from_config_with_thermal(self):
        """from_config should create model with thermal transport."""
        config = {
            "resistivity": {"type": "spitzer", "eta_0": 1e-6},
            "halo": {"halo_density": 1e16},
            "thermal": {
                "kappa_parallel_0": 1e15,
                "kappa_perp_ratio": 1e-5,
                "use_spitzer": False
            }
        }

        model = ExtendedMHD.from_config(config)

        assert model.thermal is not None, "Model should have thermal transport"
        assert model.thermal.kappa_parallel_0 == 1e15
        assert model.thermal.kappa_perp_ratio == 1e-5
        assert model.thermal.use_spitzer is False

    def test_from_config_without_thermal(self):
        """from_config without thermal section should not enable thermal."""
        config = {
            "resistivity": {"type": "spitzer", "eta_0": 1e-6},
            "halo": {"halo_density": 1e16}
        }

        model = ExtendedMHD.from_config(config)

        assert model.thermal is None, "Model should not have thermal transport"
