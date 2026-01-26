"""Tests for merging diagnostics."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.merging import MergingDiagnostics


class TestMergingDiagnostics:
    """Tests for MergingDiagnostics probe."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0
        )

    @pytest.fixture
    def two_frc_state(self, geometry):
        """Create state with two FRC-like structures."""
        state = State.zeros(nr=20, nz=40)

        # Create two peaked psi structures
        r = geometry.r_grid
        z = geometry.z_grid

        # FRC 1 centered at z=-1
        psi1 = jnp.exp(-((r - 0.5)**2 + (z + 1.0)**2) / 0.1)
        # FRC 2 centered at z=+1
        psi2 = jnp.exp(-((r - 0.5)**2 + (z - 1.0)**2) / 0.1)

        psi = psi1 + psi2

        # Set pressure proportional to psi
        p = psi * 0.5

        return state.replace(psi=psi, p=p, n=jnp.ones_like(psi))

    def test_computes_separation(self, two_frc_state, geometry):
        """Diagnostics compute null separation."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "separation_dz" in result
        # Two FRCs at z=-1 and z=+1, separation ~2.0
        assert 1.5 < result["separation_dz"] < 2.5

    def test_computes_separatrix_radius(self, two_frc_state, geometry):
        """Diagnostics compute separatrix radius."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "separatrix_radius" in result
        assert result["separatrix_radius"] > 0

    def test_computes_peak_pressure(self, two_frc_state, geometry):
        """Diagnostics compute peak pressure."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "peak_pressure" in result
        assert result["peak_pressure"] > 0

    def test_computes_elongation(self, two_frc_state, geometry):
        """Diagnostics compute elongation."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "elongation" in result
        assert result["elongation"] > 0  # Positive for elongated FRC

    def test_reconnection_rate_in_compute_result(self, two_frc_state, geometry):
        """Reconnection rate is included in compute() output."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "reconnection_rate" in result

    def test_reconnection_rate_non_negative(self, two_frc_state, geometry):
        """Reconnection rate is non-negative (uses absolute value)."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert result["reconnection_rate"] >= 0

    def test_reconnection_rate_zero_when_E_field_zero(self, geometry):
        """Reconnection rate is zero when E field is zero."""
        # Create state with zero E field
        state = State.zeros(nr=20, nz=40)
        # E field is already zeros from State.zeros()

        diag = MergingDiagnostics()
        result = diag.compute(state, geometry)

        assert result["reconnection_rate"] == 0.0
