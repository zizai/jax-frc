"""Tests for nuclear burn physics."""

import jax.numpy as jnp
import pytest


class TestBoschHaleCoefficients:
    """Tests for Bosch-Hale reactivity fit coefficients."""

    def test_dt_coefficients_exist(self):
        """D-T reaction coefficients are defined."""
        from jax_frc.constants import BOSCH_HALE_DT
        assert "C1" in BOSCH_HALE_DT
        assert "C2" in BOSCH_HALE_DT
        assert "C3" in BOSCH_HALE_DT
        assert "C4" in BOSCH_HALE_DT
        assert "C5" in BOSCH_HALE_DT
        assert "C6" in BOSCH_HALE_DT
        assert "C7" in BOSCH_HALE_DT

    def test_dd_coefficients_exist(self):
        """D-D reaction coefficients are defined."""
        from jax_frc.constants import BOSCH_HALE_DD_T, BOSCH_HALE_DD_HE3
        assert "C1" in BOSCH_HALE_DD_T
        assert "C1" in BOSCH_HALE_DD_HE3

    def test_dhe3_coefficients_exist(self):
        """D-3He reaction coefficients are defined."""
        from jax_frc.constants import BOSCH_HALE_DHE3
        assert "C1" in BOSCH_HALE_DHE3


class TestReactivity:
    """Tests for reactivity <sigma*v> calculations."""

    def test_reactivity_importable(self):
        """Reactivity function is importable."""
        from jax_frc.burn.physics import reactivity
        assert callable(reactivity)

    def test_dt_reactivity_at_10kev(self):
        """D-T reactivity at 10 keV matches published value."""
        from jax_frc.burn.physics import reactivity
        # Published value: ~1.1e-16 cm³/s at 10 keV
        sigma_v = reactivity(10.0, "DT")
        assert 1.0e-16 < sigma_v < 1.3e-16

    def test_dt_reactivity_at_20kev(self):
        """D-T reactivity at 20 keV matches published value."""
        from jax_frc.burn.physics import reactivity
        # Published value: ~4.2e-16 cm³/s at 20 keV
        sigma_v = reactivity(20.0, "DT")
        assert 3.5e-16 < sigma_v < 5.0e-16

    def test_dt_reactivity_peak(self):
        """D-T reactivity peaks around 64 keV."""
        from jax_frc.burn.physics import reactivity
        sigma_v_50 = reactivity(50.0, "DT")
        sigma_v_64 = reactivity(64.0, "DT")
        sigma_v_80 = reactivity(80.0, "DT")
        assert sigma_v_64 > sigma_v_50
        assert sigma_v_64 > sigma_v_80

    def test_dd_reactivity_lower_than_dt(self):
        """D-D reactivity is lower than D-T at same temperature."""
        from jax_frc.burn.physics import reactivity
        sigma_v_dt = reactivity(10.0, "DT")
        sigma_v_dd = reactivity(10.0, "DD_T") + reactivity(10.0, "DD_HE3")
        assert sigma_v_dd < sigma_v_dt

    def test_reactivity_vectorized(self):
        """Reactivity works with JAX arrays."""
        from jax_frc.burn.physics import reactivity
        T = jnp.array([5.0, 10.0, 20.0])
        sigma_v = reactivity(T, "DT")
        assert sigma_v.shape == (3,)
        assert jnp.all(sigma_v > 0)