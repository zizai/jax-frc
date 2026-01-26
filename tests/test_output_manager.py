"""Tests for OutputManager."""

import pytest
import tempfile
from pathlib import Path
import shutil

from jax_frc.diagnostics.output import OutputManager


class TestOutputManager:
    """Test OutputManager file operations."""

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_creates_output_directory(self, output_dir):
        """OutputManager creates directory structure."""
        manager = OutputManager(
            output_dir=output_dir / "test_run",
            example_name="belova_case2",
        )
        manager.setup()
        assert manager.run_dir.exists()
        assert (manager.run_dir / "plots").exists()

    def test_save_history_csv(self, output_dir):
        """save_history creates CSV file."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0, 1.0, 2.0], "B_max": [0.1, 0.2, 0.3]}
        manager.save_history(history, format="csv")
        assert (manager.run_dir / "history.csv").exists()

    def test_save_history_json(self, output_dir):
        """save_history creates JSON file."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0, 1.0], "psi_max": [1.0, 0.9]}
        manager.save_history(history, format="json")
        assert (manager.run_dir / "history.json").exists()

    def test_save_config_copy(self, output_dir):
        """save_config copies config to output."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        config = {"configuration": {"class": "TestConfig"}, "runtime": {"dt": 1e-6}}
        manager.save_config(config)
        assert (manager.run_dir / "config.yaml").exists()

    def test_get_summary(self, output_dir):
        """get_summary returns dict of output paths."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0], "B_max": [0.1]}
        manager.save_history(history)
        summary = manager.get_summary()
        assert "run_dir" in summary
        assert "history" in summary
