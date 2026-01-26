# tests/test_atomic_rates.py
"""Tests for atomic rate coefficients."""

import jax.numpy as jnp
import pytest

from jax_frc.constants import QE


class TestIonizationRateCoefficient:
    """Tests for hydrogen ionization rate coefficient."""

    def test_ionization_rate_coefficient_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        assert callable(ionization_rate_coefficient)

    def test_ionization_rate_positive(self):
        """Rate coefficient is positive for physical temperatures."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te = jnp.array([10.0, 50.0, 100.0, 500.0]) * QE  # eV to Joules
        sigma_v = ionization_rate_coefficient(Te)
        assert jnp.all(sigma_v > 0)

    def test_ionization_rate_peak_location(self):
        """Ionization peaks around 50-100 eV for hydrogen."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te_eV = jnp.logspace(0, 3, 100)  # 1 eV to 1 keV
        Te = Te_eV * QE
        sigma_v = ionization_rate_coefficient(Te)
        peak_idx = jnp.argmax(sigma_v)
        peak_Te_eV = Te_eV[peak_idx]
        assert 30 < peak_Te_eV < 150, f"Peak at {peak_Te_eV} eV, expected 30-150 eV"

    def test_ionization_rate_low_Te_small(self):
        """Rate is very small below ionization threshold (~13.6 eV)."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te_low = 1.0 * QE  # 1 eV
        Te_high = 100.0 * QE  # 100 eV
        assert ionization_rate_coefficient(Te_low) < 0.01 * ionization_rate_coefficient(Te_high)


class TestIonizationRate:
    """Tests for mass ionization rate S_ion."""

    def test_ionization_rate_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import ionization_rate
        assert callable(ionization_rate)

    def test_ionization_rate_dimensions(self):
        """Output has same shape as inputs."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = jnp.ones((16, 32)) * 50 * QE  # 50 eV
        ne = jnp.ones((16, 32)) * 1e19  # m^-3
        rho_n = jnp.ones((16, 32)) * 1e-6  # kg/m³
        S_ion = ionization_rate(Te, ne, rho_n)
        assert S_ion.shape == (16, 32)

    def test_ionization_rate_positive(self):
        """Rate is positive for positive inputs."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = jnp.ones((16, 32)) * 50 * QE
        ne = jnp.ones((16, 32)) * 1e19
        rho_n = jnp.ones((16, 32)) * 1e-6
        S_ion = ionization_rate(Te, ne, rho_n)
        assert jnp.all(S_ion > 0)

    def test_ionization_rate_scales_with_density(self):
        """Doubling ne or rho_n doubles rate."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = 50 * QE
        ne = 1e19
        rho_n = 1e-6
        S1 = ionization_rate(Te, ne, rho_n)
        S2 = ionization_rate(Te, 2*ne, rho_n)
        assert jnp.isclose(S2, 2*S1, rtol=1e-5)


class TestRecombinationRate:
    """Tests for radiative recombination."""

    def test_recombination_rate_coefficient_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import recombination_rate_coefficient
        assert callable(recombination_rate_coefficient)

    def test_recombination_rate_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import recombination_rate
        assert callable(recombination_rate)

    def test_recombination_decreases_with_Te(self):
        """Recombination rate decreases as Te increases."""
        from jax_frc.models.atomic_rates import recombination_rate_coefficient
        Te_low = 1.0 * QE
        Te_high = 100.0 * QE
        assert recombination_rate_coefficient(Te_low) > recombination_rate_coefficient(Te_high)

    def test_recombination_dominates_at_low_Te(self):
        """Recombination > ionization below ~1 eV."""
        from jax_frc.models.atomic_rates import (
            ionization_rate_coefficient,
            recombination_rate_coefficient,
        )
        Te_low = 1.0 * QE  # 1 eV
        assert recombination_rate_coefficient(Te_low) > ionization_rate_coefficient(Te_low)

    def test_ionization_dominates_at_high_Te(self):
        """Ionization > recombination above ~50 eV."""
        from jax_frc.models.atomic_rates import (
            ionization_rate_coefficient,
            recombination_rate_coefficient,
        )
        Te_high = 100.0 * QE  # 100 eV
        assert ionization_rate_coefficient(Te_high) > recombination_rate_coefficient(Te_high)
