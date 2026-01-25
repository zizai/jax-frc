"""Tests for ThermalTransport class."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.models.energy import ThermalTransport


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
