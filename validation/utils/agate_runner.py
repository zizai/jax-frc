# validation/utils/agate_runner.py
"""Run AGATE simulations to generate reference data."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

# Case configurations
CASE_CONFIGS = {
    "brio_wu": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.1,
        "cfl": 0.4,
        "num_snapshots": 40,
    },
    "orszag_tang": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.48,
        "cfl": 0.4,
        "slope_name": "mcbeta",
        "mcbeta": 1.3,
        "num_snapshots": 40,
    },
    "gem_reconnection": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 12.0,
        "cfl": 0.4,
        "guide_field": 0.0,
        "num_snapshots": 40,
    },
    "gem_reconnection_hall": {
        "physics": "hall_mhd",
        "hall": True,
        "end_time": 2.0,
        "cfl": 0.4,
        "guide_field": 0.0,
        "num_snapshots": 40,
    },
}


def _normalize_resolution(resolution: int | list[int] | tuple[int, ...]) -> list[int]:
    """Normalize resolution to [nx, ny, nz]."""
    if isinstance(resolution, int):
        return [resolution, resolution, 1]
    return list(resolution)


def get_expected_config(case: str, resolution: list[int] | tuple[int, ...] | int) -> dict:
    """Get expected configuration for a case.

    Args:
        case: Case name ("orszag_tang" or "gem_reconnection")
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        Configuration dictionary with snapshot_times

    Raises:
        ValueError: If case is unknown
    """
    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}. Valid cases: {list(CASE_CONFIGS.keys())}")

    resolution = _normalize_resolution(resolution)
    base_config = CASE_CONFIGS[case].copy()
    end_time = base_config["end_time"]
    num_snapshots = base_config["num_snapshots"]

    # Compute evenly-spaced snapshot times
    snapshot_times = [
        i * end_time / (num_snapshots - 1)
        for i in range(num_snapshots)
    ]

    return {
        "case": case,
        "resolution": resolution,
        "snapshot_times": snapshot_times,
        **base_config,
    }


def is_cache_valid(case: str, resolution: list[int] | tuple[int, ...] | int, output_dir: Path) -> bool:
    """Check if cached data matches current configuration.

    Args:
        case: Case name
        resolution: Grid resolution as [nx, ny, nz]
        output_dir: Directory containing cached data

    Returns:
        True if cache is valid, False otherwise
    """
    resolution = _normalize_resolution(resolution)
    res_str = f"{resolution[0]}"
    config_file = output_dir / f"{case}_{res_str}.config.yaml"
    if not config_file.exists():
        return False

    try:
        with open(config_file) as f:
            cached_config = yaml.safe_load(f)
    except Exception:
        return False

    expected_config = get_expected_config(case, resolution)

    # Compare key parameters (ignore metadata like generated_at, agate_version)
    keys_to_check = ["case", "resolution", "physics", "hall", "end_time", "num_snapshots"]
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


def _save_config(case: str, resolution: list[int] | tuple[int, ...] | int, output_dir: Path) -> None:
    """Save configuration to YAML file."""
    resolution = _normalize_resolution(resolution)
    config = get_expected_config(case, resolution)
    config["agate_version"] = _get_agate_version()
    config["generated_at"] = datetime.now(timezone.utc).isoformat()

    res_str = f"{resolution[0]}"
    config_file = output_dir / f"{case}_{res_str}.config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def _run_orszag_tang(resolution: list[int], output_dir: Path) -> None:
    """Run Orszag-Tang vortex simulation with multiple snapshots."""
    from agate.framework.scenario import OTVortex
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    resolution = _normalize_resolution(resolution)
    config = get_expected_config("orszag_tang", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = OTVortex(divClean=True, hall=False)
    roller = Roller.autodefault(
        scenario,
        ncells=resolution[0],  # AGATE uses single int for square grids
        options={"cfl": 0.4, "slopeName": "mcbeta", "mcbeta": 1.3}
    )
    roller.orient("numpy")
    roller.time = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"orszag_tang_{res_str}")
    handler.outputGrid(roller.grid)

    def _safe_time(value: object) -> float:
        return float(value) if value is not None else 0.0

    print(f"Running Orszag-Tang at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        time_value = _safe_time(roller.time)
        if i == 0:
            # Output initial state
            handler.outputState(roller.grid, roller.state, time_value, newCount=i)
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"Orszag-Tang simulation failed at t={target_time}: {e}") from e
        time_value = _safe_time(roller.time)
        handler.outputState(roller.grid, roller.state, time_value, newCount=i)


def _run_gem_reconnection(resolution: list[int], output_dir: Path) -> None:
    """Run GEM reconnection simulation with multiple snapshots."""
    from agate.framework.scenario import ReconnectionGEM as GEMReconnection
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    resolution = _normalize_resolution(resolution)
    config = get_expected_config("gem_reconnection", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = GEMReconnection(divClean=True, hall=False, guide_field=0.0)
    roller = Roller.autodefault(
        scenario,
        ncells=[resolution[0], resolution[1], 1],
        options={"cfl": 0.4}
    )
    roller.orient("numpy")
    roller.time = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"gem_reconnection_{res_str}")
    handler.outputGrid(roller.grid)

    def _safe_time(value: object) -> float:
        return float(value) if value is not None else 0.0

    print(f"Running GEM reconnection at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        time_value = _safe_time(roller.time)
        if i == 0:
            handler.outputState(roller.grid, roller.state, time_value, newCount=i)
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"GEM simulation failed at t={target_time}: {e}") from e
        time_value = _safe_time(roller.time)
        handler.outputState(roller.grid, roller.state, time_value, newCount=i)


def _run_gem_reconnection_hall(resolution: list[int], output_dir: Path) -> None:
    """Run GEM reconnection Hall-MHD simulation with multiple snapshots."""
    from agate.framework.scenario import ReconnectionGEM as GEMReconnection
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    resolution = _normalize_resolution(resolution)
    config = get_expected_config("gem_reconnection_hall", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = GEMReconnection(divClean=True, hall=True, guide_field=0.0)
    roller = Roller.autodefault(
        scenario,
        ncells=[resolution[0], resolution[1], 1],
        options={"cfl": 0.4}
    )
    roller.orient("numpy")
    roller.time = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"gem_reconnection_hall_{res_str}")
    handler.outputGrid(roller.grid)

    def _safe_time(value: object) -> float:
        return float(value) if value is not None else 0.0

    print(f"Running GEM reconnection (Hall) at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        time_value = _safe_time(roller.time)
        if i == 0:
            handler.outputState(roller.grid, roller.state, time_value, newCount=i)
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"GEM Hall simulation failed at t={target_time}: {e}") from e
        time_value = _safe_time(roller.time)
        handler.outputState(roller.grid, roller.state, time_value, newCount=i)


def _run_brio_wu(resolution: list[int], output_dir: Path) -> None:
    """Run Brio-Wu shock tube with multiple snapshots."""
    from agate.framework.scenario import BrioWu
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    resolution = _normalize_resolution(resolution)
    config = get_expected_config("brio_wu", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = BrioWu()
    roller = Roller.autodefault(
        scenario,
        ncells=resolution[0],
        options={"cfl": 0.4},
    )
    roller.orient("numpy")
    roller.time = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"brio_wu_{res_str}")
    handler.outputGrid(roller.grid)

    def _safe_time(value: object) -> float:
        return float(value) if value is not None else 0.0

    print(f"Running Brio-Wu at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        time_value = _safe_time(roller.time)
        if i == 0:
            handler.outputState(roller.grid, roller.state, time_value, newCount=i)
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"Brio-Wu simulation failed at t={target_time}: {e}") from e
        time_value = _safe_time(roller.time)
        handler.outputState(roller.grid, roller.state, time_value, newCount=i)


def run_agate_simulation(
    case: Literal["brio_wu", "orszag_tang", "gem_reconnection", "gem_reconnection_hall"],
    resolution: list[int] | tuple[int, ...] | int,
    output_dir: Path,
    overwrite: bool = False
) -> Path:
    """Run AGATE simulation and save results.

    Args:
        case: Which test case to run
        resolution: Grid resolution as [nx, ny, nz]
        output_dir: Where to save HDF5 output files
        overwrite: If True, regenerate even if files exist

    Returns:
        Path to the output directory

    Raises:
        RuntimeError: If AGATE is not installed or simulation fails
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

    resolution = _normalize_resolution(resolution)
    res_str = f"{resolution[0]}"

    # Check cache
    if not overwrite and is_cache_valid(case, resolution, output_dir):
        print(f"Using cached AGATE data for {case} at resolution {resolution}")
        return output_dir

    # Clean up any partial files before regenerating
    if output_dir.exists():
        for f in output_dir.glob(f"{case}_{res_str}.*"):
            f.unlink()

    try:
        # Run simulation
        if case == "brio_wu":
            _run_brio_wu(resolution, output_dir)
        elif case == "orszag_tang":
            _run_orszag_tang(resolution, output_dir)
        elif case == "gem_reconnection":
            _run_gem_reconnection(resolution, output_dir)
        elif case == "gem_reconnection_hall":
            _run_gem_reconnection_hall(resolution, output_dir)

        # Save config
        _save_config(case, resolution, output_dir)

    except Exception as e:
        # Clean up partial files on failure
        if output_dir.exists():
            for f in output_dir.glob(f"{case}_{res_str}.*"):
                f.unlink()
        raise RuntimeError(f"AGATE simulation failed for {case} at {resolution}: {e}") from e

    return output_dir
