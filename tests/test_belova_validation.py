# tests/test_belova_validation.py
"""Quantitative validation tests against Belova et al. (arXiv:2501.03425v1)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from merging_examples import (
    belova_case1, belova_case2, belova_case3, belova_case4
)

from jax_frc.diagnostics.merging import MergingDiagnostics


class TestBelovaCase2:
    """Small FRC merging - expect complete merge by ~5-7 tA."""

    @pytest.mark.slow
    def test_complete_merge_mhd(self):
        """MHD: separation reaches near-zero."""
        scenario = belova_case2(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] < 0.5
        assert result.total_time <= 15.0

    @pytest.mark.slow
    def test_elongation_increase(self):
        """Elongation should increase after merge."""
        scenario = belova_case2()

        # Setup phase to get two-FRC initial state
        phase = scenario.phases[0]
        two_frc_state = phase.setup(
            scenario.initial_state, scenario.geometry, {}
        )
        initial_diag = MergingDiagnostics().compute(
            two_frc_state, scenario.geometry
        )

        result = scenario.run()
        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        ratio = final_diag["elongation"] / (initial_diag["elongation"] + 1e-10)
        assert ratio > 1.2  # Should increase
