# validation/utils/agate_runner.py
"""Run AGATE simulations to generate reference data."""

from pathlib import Path
from typing import Literal

import yaml

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


def is_cache_valid(case: str, resolution: int, output_dir: Path) -> bool:
    """Check if cached data matches current configuration.

    Args:
        case: Case name
        resolution: Grid resolution
        output_dir: Directory containing cached data

    Returns:
        True if cache is valid, False otherwise
    """
    config_file = output_dir / f"{case}_{resolution}.config.yaml"
    if not config_file.exists():
        return False

    try:
        with open(config_file) as f:
            cached_config = yaml.safe_load(f)
    except Exception:
        return False

    expected_config = get_expected_config(case, resolution)

    # Compare key parameters (ignore metadata like generated_at, agate_version)
    keys_to_check = ["case", "resolution", "physics", "hall", "end_time"]
    return all(
        cached_config.get(k) == expected_config.get(k)
        for k in keys_to_check
    )
