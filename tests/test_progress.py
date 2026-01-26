"""Tests for CLI progress reporting."""

import pytest
import sys
from io import StringIO
from jax_frc.diagnostics.progress import ProgressReporter


class TestProgressReporter:
    """Test ProgressReporter output."""

    def test_report_writes_to_stderr(self):
        """report() writes progress to stderr."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(
            t_end=1.0,
            output_interval=1,
            stream=stderr_capture,
        )
        reporter.report(
            t=0.5,
            step=100,
            dt=1e-6,
            phase_name="merging",
        )
        output = stderr_capture.getvalue()
        assert "merging" in output
        assert "50" in output  # 50% progress

    def test_report_shows_percentage(self):
        """Progress percentage is calculated correctly."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.25, step=50, dt=1e-6, phase_name="test")
        output = stderr_capture.getvalue()
        assert "25" in output  # 25%

    def test_report_includes_step_count(self):
        """Step count is shown in output."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.1, step=1234, dt=1e-6, phase_name="test")
        output = stderr_capture.getvalue()
        assert "1234" in output

    def test_report_disabled_does_nothing(self):
        """When enabled=False, nothing is written."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, enabled=False, stream=stderr_capture)
        reporter.report(t=0.5, step=100, dt=1e-6, phase_name="test")
        assert stderr_capture.getvalue() == ""

    def test_finish_prints_newline(self):
        """finish() prints final newline."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.5, step=100, dt=1e-6, phase_name="test")
        reporter.finish()
        output = stderr_capture.getvalue()
        assert output.endswith("\n")
