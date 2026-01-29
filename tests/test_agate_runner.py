# tests/test_agate_runner.py
"""Tests for AGATE runner module."""

import pytest
from validation.utils.agate_runner import get_expected_config


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
