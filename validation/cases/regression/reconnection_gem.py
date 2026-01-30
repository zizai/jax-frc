"""GEM Reconnection Regression Against AGATE Reference Data.

Physics:
    Hall MHD GEM reconnection in a thin-y slab (cylindrical-style geometry).

IMPORTANT LIMITATION:
    The current JAX-FRC physics models (ResistiveMHD, ExtendedMHD) only evolve
    the magnetic field B via the induction equation. They do NOT evolve:
    - Density (n) - remains constant
    - Velocity (v) - remains constant
    - Pressure (p) - remains constant

    This is fundamentally different from AGATE's full MHD solver which evolves
    all fields. Therefore, the validation comparison is only meaningful for
    the magnetic field evolution.

Note:
    AGATE reference data is at t=12.0 (only one state file available).
    Only resolution 512 is available in AGATE reference data.
"""

from __future__ import annotations

import re
import time
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.gem_reconnection import GEMReconnectionConfiguration
from jax_frc.solvers import Solver
from validation.utils.agate_data import AgateDataLoader
from validation.utils.reporting import (
    ValidationReport,
    print_field_l2_table,
    print_scalar_metrics_table,
    print_aggregate_metrics_table,
)
from validation.utils.plots import (
    create_scalar_comparison_plot,
    create_error_threshold_plot,
    create_field_comparison_plot,
    create_timeseries_comparison_plot,
    create_field_error_evolution_plot,
)
import yaml


def load_agate_config(case: str, resolution: list[int]) -> dict:
    """Load AGATE config including snapshot_times.

    Args:
        case: Case identifier ("ot" for Orszag-Tang, "gem" for GEM)
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        Configuration dictionary with snapshot_times
    """
    # Map short names to full names
    case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
    full_case = case_map.get(case.lower(), case.lower())

    loader = AgateDataLoader()
    loader.ensure_files(case, resolution[0])
    case_dir = Path(loader.cache_dir) / full_case / str(resolution[0])
    config_path = case_dir / f"{full_case}_{resolution[0]}.config.yaml"

    # If config doesn't exist (old data format), return default
    if not config_path.exists():
        # Fallback for legacy data without config
        return {"snapshot_times": None}

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_agate_snapshot(case: str, resolution: list[int], snapshot_idx: int) -> dict:
    """Load spatial fields from a specific AGATE snapshot.

    Args:
        case: Case identifier ("ot" for Orszag-Tang, "gem" for GEM)
        resolution: Grid resolution as [nx, ny, nz]
        snapshot_idx: Index of snapshot (0 to num_snapshots-1)

    Returns:
        Dict with keys: density, momentum, magnetic_field, pressure
    """
    # Map short names to full names
    case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
    full_case = case_map.get(case.lower(), case.lower())

    loader = AgateDataLoader()
    loader.ensure_files(case, resolution[0])
    case_dir = Path(loader.cache_dir) / full_case / str(resolution[0])

    # Try new naming convention first (state_000000, state_000001, etc. - 6 digits from AGATE)
    state_path = case_dir / f"{full_case}_{resolution[0]}.state_{snapshot_idx:06d}.h5"

    if not state_path.exists():
        # Fall back to old naming convention (state_0, state_1, etc.)
        def _state_key(path: Path) -> int:
            match = re.search(r"state_(\d+)", path.name)
            return int(match.group(1)) if match else 0

        state_files = sorted(case_dir.rglob("*.state_*.h5"), key=_state_key)
        if snapshot_idx >= len(state_files):
            raise FileNotFoundError(f"Snapshot {snapshot_idx} not found in {case_dir}")
        state_path = state_files[snapshot_idx]

    with h5py.File(state_path, "r") as f:
        sub = f["subID0"]
        vec = sub["vector"][:]

    rho, p, v, B = _parse_state_vector(vec)

    # Strip ghost cells (AGATE uses 2 ghost cells on each side)
    rho = rho[2:-2, 2:-2, :]
    p = p[2:-2, 2:-2, :]
    v = v[2:-2, 2:-2, :, :]
    B = B[2:-2, 2:-2, :, :]

    # Transpose from AGATE's xy-plane to JAX's xz-plane
    rho = np.transpose(rho, (0, 2, 1))
    p = np.transpose(p, (0, 2, 1))
    v = np.transpose(v, (0, 2, 1, 3))
    B = np.transpose(B, (0, 2, 1, 3))

    # Swap velocity/B components: AGATE's vy -> JAX's vz
    v = v[..., [0, 2, 1]]
    B = B[..., [0, 2, 1]]

    mom = rho[..., None] * v

    return {
        "density": rho,
        "momentum": mom,
        "magnetic_field": B,
        "pressure": p,
    }


NAME = "reconnection_gem"
DESCRIPTION = "GEM Hall reconnection regression vs AGATE reference data"

RESOLUTIONS = ([512, 512, 1],)  # Only 512 available in AGATE reference data
QUICK_RESOLUTIONS = ([512, 512, 1],)
QUICK_NUM_SNAPSHOTS = 5  # t=0, 0.25*end, 0.5*end, 0.75*end, end


def get_quick_snapshot_times(end_time: float) -> list[float]:
    """Get 5 evenly-spaced snapshot times for quick test mode."""
    return [i * end_time / 4 for i in range(5)]


# Relaxed thresholds due to different numerical schemes
L2_ERROR_TOL = 0.20  # 20% for field L2 errors
RELATIVE_ERROR_TOL = 0.20  # 20% for scalar metrics

NGAS = 4
NMAG = 3
IRHO = 0
IMX = 1
IMY = 2
IMZ = 3


def setup_configuration(quick_test: bool, resolution: list[int]) -> dict:
    """Setup simulation configuration.

    Args:
        quick_test: If True, still run full time but compare fewer snapshots
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        Configuration dictionary with nx, nz, t_end, dt, etc.
    """
    # AGATE reference data is at t=12.0
    # AGATE grid is 512 x 256 (nx x ny in AGATE's xy-plane)
    # JAX uses xz-plane, so we need nx=512, nz=256
    # Quick test mode: still run full time (t=12.0) but compare at fewer snapshots
    return {
        "nx": resolution[0],
        "nz": resolution[0] // 2,  # Match AGATE aspect ratio
        "t_end": 12.0,  # Full time for snapshot comparison
        "dt": 1e-3,
        "lambda_": 0.5,
        "psi1": 0.1,
        "B0": 1.0,
        "n0": 1.0,
        "n_b": 0.2,
    }


def compute_curl(vec: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Compute curl of a vector field, handling pseudo-2D cases."""
    vx = vec[..., 0]
    vy = vec[..., 1]
    vz = vec[..., 2]

    # Handle pseudo-2D: skip gradients in dimensions with size 1
    nx, ny, nz = vec.shape[:3]

    if ny > 1:
        dvz_dy = np.gradient(vz, dy, axis=1)
        dvx_dy = np.gradient(vx, dy, axis=1)
    else:
        dvz_dy = np.zeros_like(vz)
        dvx_dy = np.zeros_like(vx)

    if nz > 1:
        dvy_dz = np.gradient(vy, dz, axis=2)
        dvx_dz = np.gradient(vx, dz, axis=2)
    else:
        dvy_dz = np.zeros_like(vy)
        dvx_dz = np.zeros_like(vx)

    dvz_dx = np.gradient(vz, dx, axis=0)
    dvy_dx = np.gradient(vy, dx, axis=0)

    curl_x = dvz_dy - dvy_dz
    curl_y = dvx_dz - dvz_dx
    curl_z = dvy_dx - dvx_dy
    return np.stack([curl_x, curl_y, curl_z], axis=-1)


def compute_metrics(rho, p, v, B, dx: float, dy: float, dz: float) -> dict:
    """Compute normalized metrics for comparison across different setups.

    Returns energy fractions (kinetic/total, magnetic/total) which are
    dimensionless and independent of domain size and normalization.
    """
    rho = np.asarray(rho)
    p = np.asarray(p)
    v = np.asarray(v)
    B = np.asarray(B)
    v_sq = np.sum(v**2, axis=-1)
    B_sq = np.sum(B**2, axis=-1)

    kinetic = 0.5 * rho * v_sq
    magnetic = 0.5 * B_sq
    gamma = 5.0 / 3.0
    thermal = p / (gamma - 1.0)

    # Use mean energy densities (independent of domain size)
    total_density = float(np.mean(kinetic + magnetic + thermal))
    kinetic_density = float(np.mean(kinetic))
    magnetic_density = float(np.mean(magnetic))

    # Energy fractions (dimensionless, independent of normalization)
    kinetic_fraction = kinetic_density / total_density if total_density > 0 else 0.0
    magnetic_fraction = magnetic_density / total_density if total_density > 0 else 0.0

    # Normalized enstrophy and current (relative to characteristic values)
    omega = compute_curl(v, dx, dy, dz)
    enstrophy_density = float(np.mean(np.sum(omega**2, axis=-1)))

    current = compute_curl(B, dx, dy, dz)
    max_j = float(np.max(np.sqrt(np.sum(current**2, axis=-1))))

    # Normalize max_current by mean |B| to make it dimensionless
    mean_B = float(np.mean(np.sqrt(B_sq)))
    normalized_max_j = max_j / mean_B if mean_B > 0 else 0.0

    return {
        "kinetic_fraction": kinetic_fraction,
        "magnetic_fraction": magnetic_fraction,
        "mean_energy_density": total_density,
        "enstrophy_density": enstrophy_density,
        "normalized_max_current": normalized_max_j,
    }


def run_simulation(cfg: dict) -> tuple:
    config = GEMReconnectionConfiguration(
        nx=cfg["nx"],
        nz=cfg["nz"],
        lambda_=cfg["lambda_"],
        psi1=cfg["psi1"],
        B0=cfg["B0"],
        n0=cfg["n0"],
        n_b=cfg["n_b"],
    )
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    # Disable divergence cleaning to avoid numerical instability
    from jax_frc.models.extended_mhd import ExtendedMHD
    model = ExtendedMHD(eta=config.eta, apply_divergence_cleaning=False)
    # Use higher damping factor for Hall term stability
    solver = Solver.create({"type": "semi_implicit", "damping_factor": 1e12})

    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)
    output_interval = max(1, n_steps // 100)

    history = {"times": [], "metrics": []}
    metrics = compute_metrics(
        state.n, state.p, state.v, state.B, geometry.dx, geometry.dy, geometry.dz
    )
    history["times"].append(0.0)
    history["metrics"].append(metrics)

    for step_idx in range(n_steps):
        state = solver.step_checked(state, dt, model, geometry)
        if (step_idx + 1) % output_interval == 0:
            current_time = (step_idx + 1) * dt
            metrics = compute_metrics(
                state.n,
                state.p,
                state.v,
                state.B,
                geometry.dx,
                geometry.dy,
                geometry.dz,
            )
            history["times"].append(current_time)
            history["metrics"].append(metrics)

    return state, geometry, history


def run_simulation_with_snapshots(
    cfg: dict,
    snapshot_times: list[float]
) -> tuple[list, any, dict]:
    """Run JAX-FRC, capturing state at each snapshot time.

    Args:
        cfg: Configuration dict
        snapshot_times: List of times to capture state

    Returns:
        Tuple of (states_list, geometry, history)
    """
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = GEMReconnectionConfiguration(
        nx=cfg["nx"],
        nz=cfg["nz"],
        lambda_=cfg["lambda_"],
        psi1=cfg["psi1"],
        B0=cfg["B0"],
        n0=cfg["n0"],
        n_b=cfg["n_b"],
    )
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = ExtendedMHD(eta=config.eta, apply_divergence_cleaning=False)
    solver = Solver.create({"type": "semi_implicit", "damping_factor": 1e12})

    dt = cfg["dt"]
    states = []
    history = {"times": [], "metrics": []}

    for target_time in snapshot_times:
        # Step until we reach target_time
        while state.time < target_time - 1e-12:
            step_dt = min(dt, target_time - state.time)
            state = solver.step_checked(state, step_dt, model, geometry)

        # Capture state at this snapshot
        states.append(state)
        metrics = compute_metrics(
            state.n, state.p, state.v, state.B,
            geometry.dx, geometry.dy, geometry.dz
        )
        history["times"].append(float(state.time))
        history["metrics"].append(metrics)

    return states, geometry, history


def _parse_state_vector(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = vec[IRHO]
    # AGATE stores momentum components in gas slots; convert to velocity for metrics.
    mom = np.stack([vec[IMX], vec[IMY], vec[IMZ]], axis=-1)
    v = np.divide(mom, rho[..., None], out=np.zeros_like(mom), where=rho[..., None] != 0)
    B = vec[NGAS : NGAS + NMAG]
    B = np.moveaxis(B, 0, -1)
    p_idx = NGAS + NMAG
    if vec.shape[0] > p_idx:
        p = vec[p_idx]
    else:
        p = np.zeros_like(rho)
    return rho, p, v, B


def load_agate_series(case: str, resolution: int) -> tuple[np.ndarray, dict]:
    # Map short names to full names
    case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
    full_case = case_map.get(case.lower(), case.lower())

    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / full_case / str(resolution)
    grid_files = list(case_dir.rglob("*.grid.h5"))
    if not grid_files:
        raise FileNotFoundError(f"No AGATE grid file found in {case_dir}")
    grid_path = grid_files[0]

    def _state_key(path: Path) -> int:
        match = re.search(r"state_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    state_files = sorted(case_dir.rglob("*.state_*.h5"), key=_state_key)
    if not state_files:
        raise FileNotFoundError(f"No AGATE state files found in {case_dir}")

    with h5py.File(grid_path, "r") as f:
        sub = f["subID0"]
        dx = float(sub.attrs.get("dx", np.diff(sub["x"][:, 0, 0]).mean()))
        dy = float(sub.attrs.get("dy", np.diff(sub["y"][0, :, 0]).mean()))
        dz = float(sub.attrs.get("dz", 1.0))
        # For 2D simulations, dz may be 0 in AGATE; use 1.0 for volume integration
        if dz == 0.0:
            dz = 1.0

    times = []
    metrics = {key: [] for key in ["kinetic_fraction", "magnetic_fraction", "mean_energy_density", "enstrophy_density", "normalized_max_current"]}
    for path in state_files:
        with h5py.File(path, "r") as f:
            sub = f["subID0"]
            vec = sub["vector"][:]
            time_attr = None
            if "otherParams" in sub:
                time_attr = sub["otherParams"].attrs.get("time")
            if time_attr is None:
                time_attr = float(len(times))
            rho, p, v, B = _parse_state_vector(vec)
            # Strip ghost cells (AGATE uses 2 ghost cells on each side)
            rho = rho[2:-2, 2:-2, :]
            p = p[2:-2, 2:-2, :]
            v = v[2:-2, 2:-2, :, :]
            B = B[2:-2, 2:-2, :, :]
            metric = compute_metrics(rho, p, v, B, dx, dy, dz)
            times.append(float(time_attr))
            for key in metrics:
                metrics[key].append(metric[key])

    return np.array(times), {k: np.array(v) for k, v in metrics.items()}


def load_agate_fields(case: str, resolution: int, use_initial: bool = False) -> dict:
    """Load spatial fields from AGATE reference data.

    Args:
        case: Case identifier ("gem" for GEM reconnection)
        resolution: Grid resolution
        use_initial: If True, load initial state; otherwise load final state

    Returns:
        Dict with keys: density, momentum, magnetic_field, pressure
        Each value is a numpy array of the spatial field.
    """
    # Map short names to full names
    case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
    full_case = case_map.get(case.lower(), case.lower())

    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / full_case / str(resolution)

    def _state_key(path: Path) -> int:
        match = re.search(r"state_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    state_files = sorted(case_dir.rglob("*.state_*.h5"), key=_state_key)
    if not state_files:
        raise FileNotFoundError(f"No AGATE state files found in {case_dir}")

    # Load initial or final state file
    state_path = state_files[0] if use_initial else state_files[-1]

    with h5py.File(state_path, "r") as f:
        sub = f["subID0"]
        vec = sub["vector"][:]

    # Parse the state vector into fields
    rho, p, v, B = _parse_state_vector(vec)

    # Strip ghost cells (AGATE uses 2 ghost cells on each side)
    # AGATE data shape: (nx+4, ny+4, 1) -> strip to (nx, ny, 1)
    rho = rho[2:-2, 2:-2, :]
    p = p[2:-2, 2:-2, :]
    v = v[2:-2, 2:-2, :, :]
    B = B[2:-2, 2:-2, :, :]

    # Transpose from AGATE's xy-plane (nx, ny, 1) to JAX's xz-plane (nx, 1, nz)
    # AGATE: x varies along axis 0, y varies along axis 1, z is singleton axis 2
    # JAX:   x varies along axis 0, y is singleton axis 1, z varies along axis 2
    # So we swap axes 1 and 2
    rho = np.transpose(rho, (0, 2, 1))
    p = np.transpose(p, (0, 2, 1))
    v = np.transpose(v, (0, 2, 1, 3))
    B = np.transpose(B, (0, 2, 1, 3))

    # Also need to swap velocity/B components: AGATE's vy -> JAX's vz, AGATE's vz -> JAX's vy
    v = v[..., [0, 2, 1]]  # Swap y and z components
    B = B[..., [0, 2, 1]]  # Swap y and z components

    # Compute momentum from density and velocity
    mom = rho[..., None] * v

    return {
        "density": rho,
        "momentum": mom,
        "magnetic_field": B,
        "pressure": p,
    }


def compute_field_l2_errors(jax_state, agate_fields: dict) -> dict:
    """Compute L2 errors between normalized JAX and AGATE spatial fields.

    Fields are normalized by their max absolute value before comparison,
    making the comparison independent of different amplitude normalizations.

    Args:
        jax_state: JAX simulation final state with n, v, B, p attributes
        agate_fields: Dict from load_agate_fields

    Returns:
        Dict of field name -> L2 error value
    """
    from jax_frc.validation.metrics import l2_error

    def normalize_field(field):
        """Normalize field to [-1, 1] range."""
        max_val = np.max(np.abs(field))
        if max_val > 1e-10:
            return field / max_val
        return field

    errors = {}

    # Density (normalize each before comparison)
    jax_rho = normalize_field(np.asarray(jax_state.n))
    agate_rho = normalize_field(agate_fields['density'])
    errors['density'] = float(l2_error(jax_rho, agate_rho))

    # Momentum (rho * v) - normalize
    jax_mom = np.asarray(jax_state.n)[..., None] * np.asarray(jax_state.v)
    jax_mom = normalize_field(jax_mom)
    agate_mom = normalize_field(agate_fields['momentum'])
    errors['momentum'] = float(l2_error(jax_mom, agate_mom))

    # Magnetic field - normalize
    jax_B = normalize_field(np.asarray(jax_state.B))
    agate_B = normalize_field(agate_fields['magnetic_field'])
    errors['magnetic_field'] = float(l2_error(jax_B, agate_B))

    # Pressure - normalize
    jax_p = normalize_field(np.asarray(jax_state.p))
    agate_p = normalize_field(agate_fields['pressure'])
    errors['pressure'] = float(l2_error(jax_p, agate_p))

    return errors


def compute_field_metrics(jax_field: np.ndarray, agate_field: np.ndarray) -> dict:
    """Compute per-field metrics: L2 error, max abs error, relative error.

    Args:
        jax_field: JAX field array
        agate_field: AGATE reference field array

    Returns:
        Dict with l2_error, max_abs_error, relative_error
    """
    from jax_frc.validation.metrics import l2_error

    # Normalize fields for comparison
    def normalize(field):
        max_val = np.max(np.abs(field))
        return field / max_val if max_val > 1e-10 else field

    jax_norm = normalize(jax_field)
    agate_norm = normalize(agate_field)

    diff = jax_norm - agate_norm
    l2 = float(l2_error(jax_norm, agate_norm))
    max_abs = float(np.max(np.abs(diff)))

    agate_mag = np.max(np.abs(agate_norm))
    rel = float(np.sqrt(np.mean(diff**2)) / agate_mag) if agate_mag > 1e-10 else 0.0

    return {
        "l2_error": l2,
        "max_abs_error": max_abs,
        "relative_error": rel,
    }


def validate_all_snapshots(
    jax_states: list,
    case: str,
    resolution: list[int],
    snapshot_times: list[float]
) -> list[dict]:
    """Compare JAX-FRC vs AGATE at each snapshot.

    Args:
        jax_states: List of JAX states at each snapshot
        case: Case name (e.g., "ot" for Orszag-Tang)
        resolution: Grid resolution as [nx, ny, nz]
        snapshot_times: List of snapshot times

    Returns:
        List of dicts with time and per-field errors
    """
    all_errors = []
    for i, (jax_state, t) in enumerate(zip(jax_states, snapshot_times)):
        agate_fields = load_agate_snapshot(case, resolution, snapshot_idx=i)

        field_errors = {}

        # Density
        jax_rho = np.asarray(jax_state.n)
        field_errors["density"] = compute_field_metrics(jax_rho, agate_fields["density"])

        # Momentum
        jax_mom = np.asarray(jax_state.n)[..., None] * np.asarray(jax_state.v)
        field_errors["momentum"] = compute_field_metrics(jax_mom, agate_fields["momentum"])

        # Magnetic field
        jax_B = np.asarray(jax_state.B)
        field_errors["magnetic_field"] = compute_field_metrics(jax_B, agate_fields["magnetic_field"])

        # Pressure
        jax_p = np.asarray(jax_state.p)
        field_errors["pressure"] = compute_field_metrics(jax_p, agate_fields["pressure"])

        all_errors.append({"time": t, "errors": field_errors})

    return all_errors


def compute_aggregate_metrics(
    jax_history: dict,
    agate_times: np.ndarray,
    agate_metrics: dict
) -> dict:
    """Compute time-series comparison metrics for aggregate quantities.

    Args:
        jax_history: Dict with 'times' and 'metrics' lists from JAX
        agate_times: Array of AGATE snapshot times
        agate_metrics: Dict of metric_name -> array of values

    Returns:
        Dict of metric_name -> {mean_residual, std_residual, relative_error}
    """
    results = {}

    # Extract JAX time-series
    jax_times = np.array(jax_history["times"])
    jax_metrics_by_key = {}
    for key in ["kinetic_fraction", "magnetic_fraction", "mean_energy_density",
                "enstrophy_density", "normalized_max_current"]:
        jax_metrics_by_key[key] = np.array([m[key] for m in jax_history["metrics"]])

    for key in jax_metrics_by_key:
        jax_vals = jax_metrics_by_key[key]
        agate_vals = agate_metrics[key]

        # Interpolate to common times if needed
        if len(jax_vals) != len(agate_vals):
            # Use AGATE times as reference
            jax_interp = np.interp(agate_times, jax_times, jax_vals)
            residuals = jax_interp - agate_vals
        else:
            residuals = jax_vals - agate_vals

        mean_resid = float(np.mean(residuals))
        std_resid = float(np.std(residuals))
        agate_norm = float(np.linalg.norm(agate_vals))
        rel_error = float(np.linalg.norm(residuals) / agate_norm) if agate_norm > 1e-10 else 0.0

        results[key] = {
            "mean_residual": mean_resid,
            "std_residual": std_resid,
            "relative_error": rel_error,
        }

    return results


def compare_final_values(jax_value: float, agate_value: float, metric_name: str) -> tuple[bool, dict]:
    """Compare final state values using relative error.

    Args:
        jax_value: Final metric value from JAX simulation
        agate_value: Final metric value from AGATE reference
        metric_name: Name of the metric being compared

    Returns:
        Tuple of (passed, stats_dict)
    """
    # Check for NaN/Inf in input values
    if np.isnan(jax_value) or np.isinf(jax_value):
        return False, {
            "jax_value": float("nan"),
            "agate_value": agate_value,
            "l2_error": float("nan"),
            "relative_error": float("nan"),
            "error_msg": "NaN or Inf detected in JAX simulation value",
        }
    if np.isnan(agate_value) or np.isinf(agate_value):
        return False, {
            "jax_value": jax_value,
            "agate_value": float("nan"),
            "l2_error": float("nan"),
            "relative_error": float("nan"),
            "error_msg": "NaN or Inf detected in AGATE reference value",
        }

    # Compute relative error: |jax - agate| / max(|agate|, 1e-8)
    denom = max(abs(agate_value), 1e-8)
    relative_error = abs(jax_value - agate_value) / denom

    # Check for NaN/Inf in computed error
    if np.isnan(relative_error) or np.isinf(relative_error):
        return False, {
            "jax_value": jax_value,
            "agate_value": agate_value,
            "l2_error": float("nan"),
            "relative_error": float("nan"),
            "error_msg": "NaN or Inf in computed error",
        }

    # Use unified relative error threshold
    threshold = RELATIVE_ERROR_TOL
    passed = relative_error <= threshold

    return passed, {
        "jax_value": jax_value,
        "agate_value": agate_value,
        "l2_error": relative_error,
        "relative_error": relative_error,
        "threshold": threshold,
        "threshold_type": "relative",
    }


def main(quick_test: bool = False) -> bool:
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    if quick_test:
        print("  (QUICK TEST MODE - 5 snapshots)")
    print()

    print("Configuration:")
    resolutions = QUICK_RESOLUTIONS if quick_test else RESOLUTIONS
    print(f"  resolutions: {resolutions}")
    print(f"  Field L2 threshold: {L2_ERROR_TOL} ({L2_ERROR_TOL*100:.0f}%)")
    print(f"  Relative error threshold: {RELATIVE_ERROR_TOL} ({RELATIVE_ERROR_TOL*100:.0f}%)")
    print()

    overall_pass = True
    all_results = {}
    all_metrics = {}

    # Download AGATE data before running simulations
    print("Downloading AGATE reference data...")
    for resolution in resolutions:
        try:
            loader = AgateDataLoader()
            loader.ensure_files("gem", resolution[0])
            print(f"  Resolution {resolution[0]}: OK")
        except Exception as exc:
            print(f"  Resolution {resolution[0]}: FAILED ({exc})")
    print()

    for resolution in resolutions:
        res_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
        print(f"Resolution {res_str}:")

        # Load AGATE config to get snapshot times
        try:
            agate_config = load_agate_config("gem", resolution)
        except Exception as exc:
            print(f"  ERROR: Failed to load AGATE config: {exc}")
            overall_pass = False
            continue

        # Determine snapshot times
        if quick_test:
            snapshot_times = get_quick_snapshot_times(12.0)
            print(f"  Using {len(snapshot_times)} snapshots (quick mode)")
        else:
            snapshot_times = agate_config.get("snapshot_times")
            if snapshot_times is None:
                # Fallback for legacy data without config
                snapshot_times = [0.0, 12.0]
            print(f"  Using {len(snapshot_times)} snapshots")

        # Run simulation with snapshots
        cfg = setup_configuration(quick_test, resolution)
        print(f"  Running simulation to t={cfg['t_end']}...", end="", flush=True)
        t_start = time.time()
        try:
            jax_states, geometry, history = run_simulation_with_snapshots(cfg, snapshot_times)
        except Exception as exc:
            t_sim = time.time() - t_start
            print(f" [{t_sim:.2f}s]")
            print(f"  ERROR: Simulation failed: {exc}")
            overall_pass = False
            continue
        t_sim = time.time() - t_start
        print(f" [{t_sim:.2f}s]")

        # Validate all snapshots
        print(f"  Validating {len(snapshot_times)} snapshots...")
        try:
            snapshot_errors = validate_all_snapshots(
                jax_states, "gem", resolution, snapshot_times
            )
        except Exception as exc:
            print(f"  ERROR: Failed to validate snapshots: {exc}")
            overall_pass = False
            continue

        # Compute aggregate metrics
        agate_times = None
        agate_scalar_metrics = None
        try:
            agate_times, agate_scalar_metrics = load_agate_series("gem", resolution[0])
            aggregate_metrics = compute_aggregate_metrics(
                history, agate_times, agate_scalar_metrics
            )
        except Exception as exc:
            print(f"  WARNING: Failed to compute aggregate metrics: {exc}")
            aggregate_metrics = {}

        # Print per-field table (use last snapshot's errors for summary)
        final_field_errors = snapshot_errors[-1]["errors"]
        print()
        print_field_l2_table(final_field_errors, L2_ERROR_TOL)
        print()

        # Print aggregate metrics table
        if aggregate_metrics:
            print_aggregate_metrics_table(aggregate_metrics, RELATIVE_ERROR_TOL)
            print()

        # Summary for this resolution
        # Count field passes (based on final snapshot L2 errors)
        field_passed = sum(
            1 for stats in final_field_errors.values()
            if stats["l2_error"] <= L2_ERROR_TOL
        )
        # Count aggregate passes
        agg_passed = sum(
            1 for stats in aggregate_metrics.values()
            if stats["relative_error"] <= RELATIVE_ERROR_TOL
        ) if aggregate_metrics else 0

        total_checks = len(final_field_errors) + len(aggregate_metrics)
        total_passed = field_passed + agg_passed
        res_pass = total_passed == total_checks
        overall_pass = overall_pass and res_pass

        print(f"  Summary: {total_passed}/{total_checks} PASS")
        print()

        # Store results for report generation
        all_results[resolution[0]] = {
            'field_errors': final_field_errors,
            'snapshot_errors': snapshot_errors,
            'aggregate_metrics': aggregate_metrics,
            'jax_states': jax_states,
            'history': history,
            'agate_times': agate_times,
            'agate_scalar_metrics': agate_scalar_metrics,
            'resolution': resolution,
        }

        # Store metrics for report
        for field, stats in final_field_errors.items():
            all_metrics[f"{field}_l2_r{resolution[0]}"] = {
                'value': stats["l2_error"],
                'threshold': L2_ERROR_TOL,
                'passed': stats["l2_error"] <= L2_ERROR_TOL,
                'description': f'{field} L2 error vs AGATE (final snapshot)',
            }
        for key, stats in aggregate_metrics.items():
            all_metrics[f"{key}_r{resolution[0]}"] = {
                'value': stats["relative_error"],
                'threshold': RELATIVE_ERROR_TOL,
                'passed': stats["relative_error"] <= RELATIVE_ERROR_TOL,
                'description': f'{key} time-series relative error vs AGATE',
            }

    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={
            "resolutions": [list(r) for r in resolutions],
            "L2_threshold": L2_ERROR_TOL,
            "relative_threshold": RELATIVE_ERROR_TOL,
            "quick_test": quick_test,
        },
        metrics=all_metrics,
        overall_pass=overall_pass,
    )

    # Generate plots for each resolution
    for res_key, data in all_results.items():
        resolution = data['resolution']

        # Plot 1: Field error evolution over time
        fig_error_evol = create_field_error_evolution_plot(
            data['snapshot_errors'], L2_ERROR_TOL, resolution
        )
        report.add_plot(
            fig_error_evol,
            name=f"field_error_evolution_r{res_key}",
            caption=f"Per-field L2 error evolution at resolution {res_key}"
        )
        plt.close(fig_error_evol)

        # Plot 2: Time-series comparison for each aggregate metric
        jax_times = np.array(data['history']['times'])
        agate_times = data['agate_times']
        for metric_name in ['kinetic_fraction', 'magnetic_fraction']:
            jax_vals = np.array([m[metric_name] for m in data['history']['metrics']])
            agate_vals = data['agate_scalar_metrics'][metric_name]
            fig_ts = create_timeseries_comparison_plot(
                jax_times, jax_vals, agate_times, agate_vals,
                metric_name, resolution
            )
            report.add_plot(
                fig_ts,
                name=f"{metric_name}_timeseries_r{res_key}",
                caption=f"{metric_name} time evolution at resolution {res_key}"
            )
            plt.close(fig_ts)

        # Plot 3: Field comparison at final time
        final_state = data['jax_states'][-1]
        try:
            agate_fields = load_agate_snapshot("gem", resolution, len(data['snapshot_errors']) - 1)
        except Exception:
            # Fallback to load_agate_fields for legacy data
            agate_fields = load_agate_fields("gem", resolution[0], use_initial=False)

        # Density comparison
        jax_density = np.asarray(final_state.n)[:, 0, :]
        agate_density = agate_fields['density'][:, 0, :]
        fig_density = create_field_comparison_plot(
            jax_density, agate_density,
            'density', res_key, data['field_errors']['density']['l2_error']
        )
        report.add_plot(
            fig_density,
            name=f"density_comparison_r{res_key}",
            caption=f"Density field comparison at resolution {res_key}"
        )
        plt.close(fig_density)

        # Magnetic field Bz comparison
        jax_bz = np.asarray(final_state.B)[:, 0, :, 2]
        agate_bz = agate_fields['magnetic_field'][:, 0, :, 2]
        fig_bz = create_field_comparison_plot(
            jax_bz, agate_bz,
            'magnetic_field_Bz', res_key, data['field_errors']['magnetic_field']['l2_error']
        )
        report.add_plot(
            fig_bz,
            name=f"magnetic_field_comparison_r{res_key}",
            caption=f"Magnetic field Bz comparison at resolution {res_key}"
        )
        plt.close(fig_bz)

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    # Final result
    if overall_pass:
        print("OVERALL: PASS (all resolutions passed)")
    else:
        print("OVERALL: FAIL (some checks failed)")

    return bool(overall_pass)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
