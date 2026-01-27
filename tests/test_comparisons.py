"""Tests for Belova comparison framework.

These tests verify the data structures and comparison logic without
running actual simulations. They follow the project convention of
testing framework components independently from physics.
"""

import pytest
import jax.numpy as jnp

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.comparisons.belova_merging import (
    MergingResult,
    ComparisonReport,
    BelovaComparisonSuite,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_geometry():
    """Create a small test geometry."""
    return Geometry(
        nx=16,
        ny=4,
        nz=32,
        x_min=0.01,
        x_max=0.5,
        y_min=0.0,
        y_max=2 * jnp.pi,
        z_min=-1.0,
        z_max=1.0,
        bc_x="neumann",
        bc_y="periodic",
        bc_z="neumann",
    )


@pytest.fixture
def sample_state():
    """Create a minimal test state."""
    return State.zeros(nx=16, ny=4, nz=32)


@pytest.fixture
def sample_mhd_result(sample_state):
    """Create a sample MHD merging result."""
    times = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    return MergingResult(
        model_type="resistive_mhd",
        times=times,
        null_separation=jnp.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.01]),
        reconnection_rate=jnp.array([0.0, 0.1, 0.3, 0.5, 0.4, 0.1]),
        E_magnetic=jnp.array([1000.0, 950.0, 850.0, 700.0, 600.0, 550.0]),
        E_kinetic=jnp.array([0.0, 30.0, 80.0, 150.0, 200.0, 220.0]),
        E_thermal=jnp.array([0.0, 20.0, 70.0, 150.0, 200.0, 230.0]),
        merge_time=0.4,
        final_state=sample_state,
    )


@pytest.fixture
def sample_hybrid_result(sample_state):
    """Create a sample hybrid kinetic merging result."""
    times = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return MergingResult(
        model_type="hybrid_kinetic",
        times=times,
        null_separation=jnp.array([1.0, 0.85, 0.6, 0.35, 0.15, 0.05, 0.01]),
        reconnection_rate=jnp.array([0.0, 0.08, 0.25, 0.45, 0.55, 0.35, 0.1]),
        E_magnetic=jnp.array([1000.0, 960.0, 880.0, 750.0, 650.0, 580.0, 540.0]),
        E_kinetic=jnp.array([0.0, 25.0, 70.0, 140.0, 190.0, 230.0, 250.0]),
        E_thermal=jnp.array([0.0, 15.0, 50.0, 110.0, 160.0, 190.0, 210.0]),
        merge_time=0.5,
        final_state=sample_state,
    )


# =============================================================================
# MergingResult Tests
# =============================================================================


class TestMergingResult:
    """Tests for MergingResult data class."""

    def test_construction_with_required_fields(self, sample_state):
        """MergingResult can be constructed with all required fields."""
        times = jnp.array([0.0, 0.1, 0.2])
        result = MergingResult(
            model_type="resistive_mhd",
            times=times,
            null_separation=jnp.array([1.0, 0.5, 0.1]),
            reconnection_rate=jnp.array([0.0, 0.2, 0.1]),
            E_magnetic=jnp.array([100.0, 90.0, 85.0]),
            E_kinetic=jnp.array([0.0, 5.0, 8.0]),
            E_thermal=jnp.array([0.0, 5.0, 7.0]),
            merge_time=0.2,
            final_state=sample_state,
        )

        assert result.model_type == "resistive_mhd"
        assert result.n_timesteps == 3
        assert result.merge_time == 0.2

    def test_construction_without_merge(self, sample_state):
        """MergingResult can be constructed with merge_time=None."""
        times = jnp.array([0.0, 0.1])
        result = MergingResult(
            model_type="hybrid_kinetic",
            times=times,
            null_separation=jnp.array([1.0, 0.9]),
            reconnection_rate=jnp.array([0.0, 0.1]),
            E_magnetic=jnp.array([100.0, 98.0]),
            E_kinetic=jnp.array([0.0, 1.0]),
            E_thermal=jnp.array([0.0, 1.0]),
            merge_time=None,
            final_state=sample_state,
        )

        assert result.merge_time is None

    def test_invalid_model_type_raises(self, sample_state):
        """MergingResult raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            MergingResult(
                model_type="invalid_model",
                times=jnp.array([0.0]),
                null_separation=jnp.array([1.0]),
                reconnection_rate=jnp.array([0.0]),
                E_magnetic=jnp.array([100.0]),
                E_kinetic=jnp.array([0.0]),
                E_thermal=jnp.array([0.0]),
                merge_time=None,
                final_state=sample_state,
            )

    def test_e_total_property(self, sample_mhd_result):
        """E_total property correctly sums energy components."""
        E_total = sample_mhd_result.E_total

        # Check at first time point
        expected = (
            float(sample_mhd_result.E_magnetic[0])
            + float(sample_mhd_result.E_kinetic[0])
            + float(sample_mhd_result.E_thermal[0])
        )
        assert float(E_total[0]) == pytest.approx(expected)

    def test_energy_fraction_at_time(self, sample_mhd_result):
        """energy_fraction_at_time returns correct fractions."""
        # At merge time (0.4), check fractions sum to 1
        fractions = sample_mhd_result.energy_fraction_at_time(0.4)

        assert "f_magnetic" in fractions
        assert "f_kinetic" in fractions
        assert "f_thermal" in fractions

        total = fractions["f_magnetic"] + fractions["f_kinetic"] + fractions["f_thermal"]
        assert total == pytest.approx(1.0)

    def test_energy_fraction_zero_energy(self, sample_state):
        """energy_fraction_at_time handles zero total energy."""
        result = MergingResult(
            model_type="resistive_mhd",
            times=jnp.array([0.0]),
            null_separation=jnp.array([1.0]),
            reconnection_rate=jnp.array([0.0]),
            E_magnetic=jnp.array([0.0]),
            E_kinetic=jnp.array([0.0]),
            E_thermal=jnp.array([0.0]),
            merge_time=None,
            final_state=sample_state,
        )

        fractions = result.energy_fraction_at_time(0.0)
        assert fractions["f_magnetic"] == 0.0
        assert fractions["f_kinetic"] == 0.0
        assert fractions["f_thermal"] == 0.0


# =============================================================================
# ComparisonReport Tests
# =============================================================================


class TestComparisonReport:
    """Tests for ComparisonReport data class."""

    def test_construction(self, sample_mhd_result, sample_hybrid_result):
        """ComparisonReport can be constructed from two results."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        assert report.mhd_result.model_type == "resistive_mhd"
        assert report.hybrid_result.model_type == "hybrid_kinetic"

    def test_wrong_model_type_mhd_raises(self, sample_hybrid_result):
        """ComparisonReport raises if mhd_result has wrong model type."""
        with pytest.raises(ValueError, match="mhd_result must have model_type='resistive_mhd'"):
            ComparisonReport(
                mhd_result=sample_hybrid_result,  # Wrong type
                hybrid_result=sample_hybrid_result,
            )

    def test_wrong_model_type_hybrid_raises(self, sample_mhd_result):
        """ComparisonReport raises if hybrid_result has wrong model type."""
        with pytest.raises(ValueError, match="hybrid_result must have model_type='hybrid_kinetic'"):
            ComparisonReport(
                mhd_result=sample_mhd_result,
                hybrid_result=sample_mhd_result,  # Wrong type
            )

    def test_merge_time_difference(self, sample_mhd_result, sample_hybrid_result):
        """merge_time_difference correctly computes difference."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        # hybrid (0.5) - mhd (0.4) = 0.1
        diff = report.merge_time_difference()
        assert diff == pytest.approx(0.1)

    def test_merge_time_difference_no_merge(self, sample_mhd_result, sample_state):
        """merge_time_difference returns nan when no merge."""
        hybrid_no_merge = MergingResult(
            model_type="hybrid_kinetic",
            times=jnp.array([0.0, 0.1]),
            null_separation=jnp.array([1.0, 0.9]),
            reconnection_rate=jnp.array([0.0, 0.1]),
            E_magnetic=jnp.array([100.0, 98.0]),
            E_kinetic=jnp.array([0.0, 1.0]),
            E_thermal=jnp.array([0.0, 1.0]),
            merge_time=None,
            final_state=sample_state,
        )

        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=hybrid_no_merge,
        )

        import math
        assert math.isnan(report.merge_time_difference())

    def test_merge_time_ratio(self, sample_mhd_result, sample_hybrid_result):
        """merge_time_ratio correctly computes ratio."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        # hybrid (0.5) / mhd (0.4) = 1.25
        ratio = report.merge_time_ratio()
        assert ratio == pytest.approx(1.25)

    def test_energy_partition_at_merge(self, sample_mhd_result, sample_hybrid_result):
        """energy_partition_at_merge returns correct structure."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        partition = report.energy_partition_at_merge()

        assert "mhd" in partition
        assert "hybrid" in partition

        # Check MHD fractions sum to 1
        mhd_sum = (
            partition["mhd"]["f_magnetic"]
            + partition["mhd"]["f_kinetic"]
            + partition["mhd"]["f_thermal"]
        )
        assert mhd_sum == pytest.approx(1.0)

        # Check hybrid fractions sum to 1
        hybrid_sum = (
            partition["hybrid"]["f_magnetic"]
            + partition["hybrid"]["f_kinetic"]
            + partition["hybrid"]["f_thermal"]
        )
        assert hybrid_sum == pytest.approx(1.0)

    def test_max_reconnection_rate_ratio(self, sample_mhd_result, sample_hybrid_result):
        """max_reconnection_rate_ratio computes correctly."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        ratio = report.max_reconnection_rate_ratio()

        # MHD max: 0.5, Hybrid max: 0.55
        expected = 0.55 / 0.5
        assert ratio == pytest.approx(expected)

    def test_summary(self, sample_mhd_result, sample_hybrid_result):
        """summary returns dict with expected keys."""
        report = ComparisonReport(
            mhd_result=sample_mhd_result,
            hybrid_result=sample_hybrid_result,
        )

        summary = report.summary()

        assert "mhd_merge_time" in summary
        assert "hybrid_merge_time" in summary
        assert "merge_time_difference" in summary
        assert "merge_time_ratio" in summary
        assert "max_reconnection_rate_ratio" in summary


# =============================================================================
# BelovaComparisonSuite Tests
# =============================================================================


class TestBelovaComparisonSuite:
    """Tests for BelovaComparisonSuite."""

    def test_default_parameters(self):
        """BelovaComparisonSuite has expected default parameters."""
        suite = BelovaComparisonSuite()

        # Check physical defaults
        assert suite.initial_separation == 2.0
        assert suite.initial_velocity == 0.0
        assert suite.frc_elongation == 3.0

        # Check grid defaults
        assert suite.nr == 64
        assert suite.nz == 128
        assert suite.ny == 4
        assert suite.r_min == 0.01
        assert suite.r_max == 0.5
        assert suite.y_min == 0.0
        assert suite.z_min == -2.0
        assert suite.z_max == 2.0

    def test_custom_parameters(self):
        """BelovaComparisonSuite can be customized."""
        suite = BelovaComparisonSuite(
            initial_separation=1.5,
            nr=32,
            nz=64,
        )

        assert suite.initial_separation == 1.5
        assert suite.nr == 32
        assert suite.nz == 64
        assert suite.ny == 4

    def test_create_geometry(self):
        """create_geometry returns valid Geometry."""
        suite = BelovaComparisonSuite(nr=32, nz=64)
        geometry = suite.create_geometry()

        assert isinstance(geometry, Geometry)
        assert geometry.nx == 32
        assert geometry.ny == 4
        assert geometry.nz == 64

    def test_collect_diagnostics(self, sample_state, sample_geometry):
        """collect_diagnostics returns expected keys."""
        suite = BelovaComparisonSuite()
        diag = suite.collect_diagnostics(sample_state, sample_geometry)

        assert "separation" in diag
        assert "reconnection_rate" in diag
        assert "E_magnetic" in diag
        assert "E_kinetic" in diag
        assert "E_thermal" in diag

    def test_detect_merge_time_found(self):
        """detect_merge_time finds merge when it occurs."""
        suite = BelovaComparisonSuite()

        times = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4])
        separations = jnp.array([1.0, 0.5, 0.1, 0.005, 0.001])

        merge_time = suite.detect_merge_time(times, separations, threshold=0.01)
        assert merge_time == pytest.approx(0.3)

    def test_detect_merge_time_not_found(self):
        """detect_merge_time returns None when no merge."""
        suite = BelovaComparisonSuite()

        times = jnp.array([0.0, 0.1, 0.2])
        separations = jnp.array([1.0, 0.8, 0.5])

        merge_time = suite.detect_merge_time(times, separations, threshold=0.01)
        assert merge_time is None

    def test_compare_creates_report(self, sample_mhd_result, sample_hybrid_result):
        """compare method creates valid ComparisonReport."""
        suite = BelovaComparisonSuite()

        report = suite.compare(sample_mhd_result, sample_hybrid_result)

        assert isinstance(report, ComparisonReport)
        assert report.mhd_result is sample_mhd_result
        assert report.hybrid_result is sample_hybrid_result


# =============================================================================
# Integration Tests (Framework Only)
# =============================================================================


class TestComparisonsImport:
    """Test that comparisons module imports correctly."""

    def test_import_from_init(self):
        """Can import main classes from comparisons package."""
        from jax_frc.comparisons import (
            MergingResult,
            ComparisonReport,
            BelovaComparisonSuite,
        )

        assert MergingResult is not None
        assert ComparisonReport is not None
        assert BelovaComparisonSuite is not None
