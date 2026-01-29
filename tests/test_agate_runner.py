# tests/test_agate_runner.py
"""Tests for AGATE runner module."""

import tempfile
import yaml
from pathlib import Path

import pytest
from validation.utils.agate_runner import get_expected_config, is_cache_valid, run_agate_simulation


def test_get_expected_config_orszag_tang():
    """OT config should have correct parameters."""
    config = get_expected_config("orszag_tang", 256)

    assert config["case"] == "orszag_tang"
    assert config["resolution"] == 256
    assert config["physics"] == "ideal_mhd"
    assert config["hall"] is False
    assert config["end_time"] == 0.48
    assert config["cfl"] == 0.4


def test_get_expected_config_gem():
    """GEM config should have Hall MHD parameters."""
    config = get_expected_config("gem_reconnection", 512)

    assert config["case"] == "gem_reconnection"
    assert config["resolution"] == 512
    assert config["physics"] == "hall_mhd"
    assert config["hall"] is True
    assert config["end_time"] == 12.0


def test_get_expected_config_unknown_case():
    """Unknown case should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown case"):
        get_expected_config("unknown_case", 256)


def test_is_cache_valid_missing_config():
    """Missing config.yaml should return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        assert is_cache_valid("orszag_tang", 256, output_dir) is False


def test_is_cache_valid_matching_config():
    """Matching config should return True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        config_file = output_dir / "orszag_tang_256.config.yaml"

        # Write matching config
        config = get_expected_config("orszag_tang", 256)
        config["agate_version"] = "1.0.0"
        config["generated_at"] = "2026-01-30T00:00:00Z"

        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        assert is_cache_valid("orszag_tang", 256, output_dir) is True


def test_is_cache_valid_mismatched_resolution():
    """Mismatched resolution should return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        config_file = output_dir / "orszag_tang_256.config.yaml"

        # Write config with wrong resolution
        config = get_expected_config("orszag_tang", 512)  # Wrong resolution
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        assert is_cache_valid("orszag_tang", 256, output_dir) is False


@pytest.mark.slow
def test_run_orszag_tang_generates_files():
    """OT simulation should produce expected output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        run_agate_simulation("orszag_tang", 64, output_dir)  # Small resolution for speed

        # Check expected files exist
        assert (output_dir / "orszag_tang_64.grid.h5").exists()
        assert (output_dir / "orszag_tang_64.config.yaml").exists()

        # Check at least one state file exists
        state_files = list(output_dir.glob("orszag_tang_64.state_*.h5"))
        assert len(state_files) >= 1


@pytest.mark.slow
def test_run_gem_generates_files():
    """GEM simulation should produce expected output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        run_agate_simulation("gem_reconnection", 64, output_dir)  # Small resolution

        assert (output_dir / "gem_reconnection_64.grid.h5").exists()
        assert (output_dir / "gem_reconnection_64.config.yaml").exists()

        state_files = list(output_dir.glob("gem_reconnection_64.state_*.h5"))
        assert len(state_files) >= 1
