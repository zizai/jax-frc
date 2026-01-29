# validation/utils/agate_runner.py
"""Run AGATE simulations to generate reference data."""

from datetime import datetime, timezone
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


def _get_agate_version() -> str:
    """Get AGATE version string."""
    try:
        import agate
        return getattr(agate, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def _save_config(case: str, resolution: int, output_dir: Path) -> None:
    """Save configuration to YAML file."""
    config = get_expected_config(case, resolution)
    config["agate_version"] = _get_agate_version()
    config["generated_at"] = datetime.now(timezone.utc).isoformat()

    config_file = output_dir / f"{case}_{resolution}.config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def _run_orszag_tang(resolution: int, output_dir: Path) -> None:
    """Run Orszag-Tang vortex simulation."""
    from agate.framework.scenario import OTVortex
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    scenario = OTVortex(divClean=True, hall=False)
    roller = Roller.autodefault(
        scenario,
        ncells=resolution,
        options={"cfl": 0.4, "slopeName": "mcbeta", "mcbeta": 1.3}
    )
    roller.orient("numpy")

    print(f"Running Orszag-Tang at resolution {resolution}...")
    try:
        roller.roll(start_time=0.0, end_time=0.48, add_stopWatch=True)
    except Exception as e:
        raise RuntimeError(f"Orszag-Tang simulation failed: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)
    handler = fileHandler(directory=str(output_dir), prefix=f"orszag_tang_{resolution}")
    handler.outputGrid(roller.grid)
    handler.outputState(roller.grid, roller.state, roller.time)


def _run_gem_reconnection(resolution: int, output_dir: Path) -> None:
    """Run GEM reconnection simulation (placeholder for Task 4)."""
    raise NotImplementedError("GEM reconnection runner will be added in Task 4")


def run_agate_simulation(
    case: Literal["orszag_tang", "gem_reconnection"],
    resolution: int,
    output_dir: Path,
    overwrite: bool = False
) -> Path:
    """Run AGATE simulation and save results.

    Args:
        case: Which test case to run
        resolution: Grid resolution (e.g., 256, 512)
        output_dir: Where to save HDF5 output files
        overwrite: If True, regenerate even if files exist

    Returns:
        Path to the output directory

    Raises:
        RuntimeError: If AGATE is not installed
        ValueError: If case is unknown
    """
    try:
        import agate
    except ImportError:
        raise RuntimeError(
            "AGATE not installed. Install with: pip install -e ../agate-open-source"
        )

    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}")

    # Check cache
    if not overwrite and is_cache_valid(case, resolution, output_dir):
        print(f"Using cached AGATE data for {case} at resolution {resolution}")
        return output_dir

    # Run simulation
    if case == "orszag_tang":
        _run_orszag_tang(resolution, output_dir)
    elif case == "gem_reconnection":
        _run_gem_reconnection(resolution, output_dir)

    # Save config
    _save_config(case, resolution, output_dir)

    return output_dir
