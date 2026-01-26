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
