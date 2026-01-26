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


class TestChargeExchangeRates:
    """Tests for charge exchange momentum and energy transfer."""

    def test_charge_exchange_rates_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        assert callable(charge_exchange_rates)

    def test_charge_exchange_returns_tuple(self):
        """Returns (R_cx, Q_cx) tuple."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v_i = jnp.zeros((16, 32, 3))
        v_n = jnp.zeros((16, 32, 3))
        R_cx, Q_cx = charge_exchange_rates(Ti, ni, nn, v_i, v_n)
        assert R_cx.shape == (16, 32, 3)
        assert Q_cx.shape == (16, 32)

    def test_charge_exchange_momentum_zero_when_velocities_equal(self):
        """R_cx = 0 when v_i = v_n."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v = jnp.ones((16, 32, 3)) * 1000  # Same velocity
        R_cx, _ = charge_exchange_rates(Ti, ni, nn, v, v)
        assert jnp.allclose(R_cx, 0, atol=1e-20)

    def test_charge_exchange_energy_positive(self):
        """Q_cx > 0 for hot ions (energy flows from ions to neutrals)."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE  # Hot ions
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v_i = jnp.zeros((16, 32, 3))
        v_n = jnp.zeros((16, 32, 3))
        _, Q_cx = charge_exchange_rates(Ti, ni, nn, v_i, v_n)
        assert jnp.all(Q_cx > 0)


class TestRadiationLoss:
    """Tests for radiation loss terms."""

    def test_bremsstrahlung_loss_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        assert callable(bremsstrahlung_loss)

    def test_bremsstrahlung_positive(self):
        """Bremsstrahlung is always positive (energy sink)."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        Te = jnp.ones((16, 32)) * 100 * QE
        ne = jnp.ones((16, 32)) * 1e19
        ni = jnp.ones((16, 32)) * 1e19
        P_brem = bremsstrahlung_loss(Te, ne, ni)
        assert jnp.all(P_brem > 0)

    def test_bremsstrahlung_scales_with_density_squared(self):
        """P_brem ~ ne * ni."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        Te = 100 * QE
        ne = 1e19
        ni = 1e19
        P1 = bremsstrahlung_loss(Te, ne, ni)
        P2 = bremsstrahlung_loss(Te, 2*ne, 2*ni)
        assert jnp.isclose(P2, 4*P1, rtol=1e-5)

    def test_total_radiation_loss_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import total_radiation_loss
        assert callable(total_radiation_loss)

    def test_total_radiation_positive(self):
        """Total radiation is always positive."""
        from jax_frc.models.atomic_rates import total_radiation_loss
        Te = jnp.ones((16, 32)) * 100 * QE
        ne = jnp.ones((16, 32)) * 1e19
        ni = jnp.ones((16, 32)) * 1e19
        n_imp = jnp.ones((16, 32)) * 1e17  # 1% impurity
        S_ion = jnp.ones((16, 32)) * 1e10  # kg/m³/s
        P_rad = total_radiation_loss(Te, ne, ni, n_imp, S_ion)
        assert jnp.all(P_rad > 0)
