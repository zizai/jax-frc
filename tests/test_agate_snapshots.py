"""Tests for AGATE snapshot functionality."""

import pytest
import numpy as np


class TestSnapshotTimesComputation:
    """Tests for snapshot_times computation in AGATE runner."""

    def test_snapshot_times_evenly_spaced_orszag_tang(self):
        """Verify snapshot times are evenly distributed for Orszag-Tang."""
        from validation.utils.agate_runner import get_expected_config

        config = get_expected_config("orszag_tang", [256, 256, 1])
        times = config["snapshot_times"]

        assert len(times) == 40
        assert times[0] == 0.0
        assert abs(times[-1] - 0.48) < 1e-10

        # Check even spacing
        dt = times[1] - times[0]
        for i in range(1, len(times)):
            assert abs(times[i] - times[i - 1] - dt) < 1e-10

    def test_snapshot_times_evenly_spaced_gem(self):
        """Verify snapshot times are evenly distributed for GEM."""
        from validation.utils.agate_runner import get_expected_config

        config = get_expected_config("gem_reconnection", [128, 64, 1])
        times = config["snapshot_times"]

        assert len(times) == 40
        assert times[0] == 0.0
        assert abs(times[-1] - 12.0) < 1e-10

        # Check even spacing
        dt = times[1] - times[0]
        for i in range(1, len(times)):
            assert abs(times[i] - times[i - 1] - dt) < 1e-10

    def test_config_includes_num_snapshots(self):
        """Verify config includes num_snapshots."""
        from validation.utils.agate_runner import get_expected_config

        config = get_expected_config("orszag_tang", [256, 256, 1])
        assert "num_snapshots" in config
        assert config["num_snapshots"] == 40


class TestQuickSnapshotTimes:
    """Tests for quick mode snapshot times."""

    def test_quick_snapshot_times_orszag_tang(self):
        """Verify quick mode uses 5 snapshots for Orszag-Tang."""
        from validation.cases.regression.orszag_tang import get_quick_snapshot_times

        times = get_quick_snapshot_times(0.48)
        assert len(times) == 5
        assert times[0] == 0.0
        assert times[-1] == 0.48
        assert abs(times[2] - 0.24) < 1e-10  # midpoint


class TestFieldMetrics:
    """Tests for per-field metrics computation."""

    def test_compute_field_metrics_identical_fields(self):
        """Verify zero error for identical fields."""
        from validation.cases.regression.orszag_tang import compute_field_metrics

        field = np.random.rand(32, 32, 1)
        metrics = compute_field_metrics(field, field.copy())

        assert metrics["l2_error"] < 1e-10
        assert metrics["max_abs_error"] < 1e-10
        assert metrics["relative_error"] < 1e-10

    def test_compute_field_metrics_different_fields(self):
        """Verify non-zero error for different fields."""
        from validation.cases.regression.orszag_tang import compute_field_metrics

        # Use non-uniform fields to avoid normalization collapsing differences
        # (uniform fields normalize to the same value regardless of scale)
        np.random.seed(42)
        field1 = np.random.rand(32, 32, 1)
        field2 = field1 + 0.1 * np.random.rand(32, 32, 1)  # Add noise

        metrics = compute_field_metrics(field1, field2)

        assert metrics["l2_error"] > 0
        assert metrics["max_abs_error"] > 0
        assert metrics["relative_error"] > 0


class TestAggregateMetrics:
    """Tests for aggregate time-series metrics."""

    def test_compute_aggregate_metrics_identical(self):
        """Verify zero residuals for identical time-series."""
        from validation.cases.regression.orszag_tang import compute_aggregate_metrics

        times = np.linspace(0, 1, 10)
        values = np.sin(times)

        jax_history = {
            "times": list(times),
            "metrics": [
                {
                    "kinetic_fraction": v,
                    "magnetic_fraction": v,
                    "mean_energy_density": v,
                    "enstrophy_density": v,
                    "normalized_max_current": v,
                }
                for v in values
            ],
        }

        agate_metrics = {
            "kinetic_fraction": values,
            "magnetic_fraction": values,
            "mean_energy_density": values,
            "enstrophy_density": values,
            "normalized_max_current": values,
        }

        result = compute_aggregate_metrics(jax_history, times, agate_metrics)

        for key in result:
            assert abs(result[key]["mean_residual"]) < 1e-10
            assert abs(result[key]["std_residual"]) < 1e-10
            assert abs(result[key]["relative_error"]) < 1e-10
