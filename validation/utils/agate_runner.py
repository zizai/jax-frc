# validation/utils/agate_runner.py
"""Run AGATE simulations to generate reference data."""

from pathlib import Path
from typing import Literal

# Case configurations
CASE_CONFIGS = {
    "orszag_tang": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.48,
        "cfl": 0.4,
        "slope_name": "mcbeta",
        "mcbeta": 1.3,
    },
    "gem_reconnection": {
        "physics": "hall_mhd",
        "hall": True,
        "end_time": 12.0,
        "cfl": 0.4,
        "guide_field": 0.0,
    },
}


def get_expected_config(case: str, resolution: int) -> dict:
    """Get expected configuration for a case.

    Args:
        case: Case name ("orszag_tang" or "gem_reconnection")
        resolution: Grid resolution

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If case is unknown
    """
    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}. Valid cases: {list(CASE_CONFIGS.keys())}")

    base_config = CASE_CONFIGS[case].copy()
    return {
        "case": case,
        "resolution": resolution,
        **base_config,
    }
