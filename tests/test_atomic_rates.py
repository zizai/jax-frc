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
