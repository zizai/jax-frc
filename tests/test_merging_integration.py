# tests/test_merging_integration.py
"""Integration tests for merging scenario."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from merging_examples import belova_case2, create_default_geometry, create_initial_frc

from jax_frc.diagnostics.merging import MergingDiagnostics
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import timeout


class TestMergingIntegration:
    """Integration tests for full merging workflow."""

    def test_scenario_runs_to_completion(self):
        """Full scenario runs without errors."""
        scenario = belova_case2()
        result = scenario.run()

        assert result.success
        assert len(result.phase_results) == 1
        assert result.total_time > 0

    def test_diagnostics_track_merging(self):
        """Diagnostics correctly track merging progress."""
        geometry = create_default_geometry(rc=1.0, zc=2.0, nr=32, nz=64)
        initial = create_initial_frc(geometry, elongation=1.5, xs=0.5)

        diag = MergingDiagnostics()
        result = diag.compute(initial, geometry)

        # Should have valid metrics
        assert result["separation_dz"] >= 0
        assert result["separatrix_radius"] > 0
        assert result["peak_pressure"] >= 0

    def test_two_frc_setup_doubles_flux(self):
        """MergingPhase setup creates two FRCs with roughly double flux."""
        geometry = create_default_geometry(rc=1.0, zc=2.0, nr=32, nz=64)
        single_frc = create_initial_frc(geometry, elongation=1.5, xs=0.5)

        phase = MergingPhase(
            name="test",
            transition=timeout(1.0),
            separation=1.0,
            initial_velocity=0.1,
        )

        two_frc = phase.setup(single_frc, geometry, {})

        # Two-FRC should have roughly double the total flux
        single_flux = jnp.sum(jnp.abs(single_frc.psi))
        double_flux = jnp.sum(jnp.abs(two_frc.psi))

        assert 1.5 < double_flux / single_flux < 2.5
