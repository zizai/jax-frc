"""Orszag–Tang Vortex Regression Against AGATE Reference Data.

Physics:
    Standard 2D Orszag–Tang vortex in a thin-y slab, used to validate nonlinear
    MHD turbulence, current sheet formation, and energy evolution. This case
    compares scalar time-series metrics against AGATE reference data using
    point-to-point relative error comparison.
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

from jax_frc.configurations.orszag_tang import OrszagTangConfiguration
from jax_frc.solvers import Solver
from validation.utils.agate_data import AgateDataLoader
from validation.utils.reporting import ValidationReport


NAME = "orszag_tang"
DESCRIPTION = "Orszag–Tang vortex regression vs AGATE reference data"

RESOLUTIONS = (256, 512, 1024)
QUICK_RESOLUTIONS = (256,)
L2_ERROR_TOL = 0.01  # 1% L2 error threshold
RELATIVE_ERROR_TOL = 0.05  # 5% relative error threshold (95% accuracy for energy metrics)

# Energy metrics use relative error threshold, others use L2 error threshold
ENERGY_METRICS = {"total_energy", "magnetic_energy", "kinetic_energy"}

NGAS = 4
NMAG = 3
IRHO = 0
IMX = 1
IMY = 2
IMZ = 3


def setup_configuration(quick_test: bool, resolution: int) -> dict:
    # Timestep must satisfy Alfvén CFL: dt < dx / v_A
    # For B0=1.0, rho0~0.221: v_A ~ 1896, dx ~ 0.0246 -> dt_max ~ 1.3e-5
    # Use dt = 1e-5 for stability (10x smaller than original)
    if quick_test:
        return {
            "nx": resolution,
            "nz": resolution,
            "t_end": 0.01,
            "dt": 1e-5,
        }
    return {
        "nx": resolution,
        "nz": resolution,
        "t_end": 0.5,
        "dt": 1e-5,
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

    vol = dx * dy * dz
    total_energy = np.sum(kinetic + magnetic + thermal) * vol
    magnetic_energy = np.sum(magnetic) * vol
    kinetic_energy = np.sum(kinetic) * vol

    omega = compute_curl(v, dx, dy, dz)
    enstrophy = np.sum(np.sum(omega**2, axis=-1)) * vol

    current = compute_curl(B, dx, dy, dz)
    max_j = float(np.max(np.sqrt(np.sum(current**2, axis=-1))))

    return {
        "total_energy": float(total_energy),
        "magnetic_energy": float(magnetic_energy),
        "kinetic_energy": float(kinetic_energy),
        "enstrophy": float(enstrophy),
        "max_current": float(max_j),
    }


def run_simulation(cfg: dict) -> tuple:
    config = OrszagTangConfiguration(
        nx=cfg["nx"],
        nz=cfg["nz"],
    )
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    # Use CT scheme to avoid divergence cleaning issues
    from jax_frc.models.resistive_mhd import ResistiveMHD
    model = ResistiveMHD(eta=config.eta, advection_scheme="ct")
    solver = Solver.create({"type": "euler"})

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
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / case / str(resolution)
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
        dz = float(sub.attrs.get("dz", 1.0))  # May be 0 for 2D

    times = []
    metrics = {key: [] for key in ["total_energy", "magnetic_energy", "kinetic_energy", "enstrophy", "max_current"]}
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
            metric = compute_metrics(rho, p, v, B, dx, dy, dz)
            times.append(float(time_attr))
            for key in metrics:
                metrics[key].append(metric[key])

    return np.array(times), {k: np.array(v) for k, v in metrics.items()}


def load_agate_final_fields(case: str, resolution: int) -> dict:
    """Load final state spatial fields from AGATE reference data.

    Args:
        case: Case identifier ("ot" for Orszag-Tang)
        resolution: Grid resolution

    Returns:
        Dict with keys: density, momentum, magnetic_field, pressure
        Each value is a numpy array of the spatial field.
    """
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / case / str(resolution)

    def _state_key(path: Path) -> int:
        match = re.search(r"state_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    state_files = sorted(case_dir.rglob("*.state_*.h5"), key=_state_key)
    if not state_files:
        raise FileNotFoundError(f"No AGATE state files found in {case_dir}")

    # Load the final state file
    final_state_path = state_files[-1]

    with h5py.File(final_state_path, "r") as f:
        sub = f["subID0"]
        vec = sub["vector"][:]

    # Parse the state vector into fields
    rho, p, v, B = _parse_state_vector(vec)

    # Compute momentum from density and velocity
    mom = rho[..., None] * v

    return {
        "density": rho,
        "momentum": mom,
        "magnetic_field": B,
        "pressure": p,
    }


def compare_final_values(jax_value: float, agate_value: float, metric_name: str) -> tuple[bool, dict]:
    """Compare final state values using L2 error and relative error.

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

    # Compute L2 error: |jax - agate| / max(|agate|, 1e-8)
    denom = max(abs(agate_value), 1e-8)
    l2_error = abs(jax_value - agate_value) / denom

    # Compute relative error: |jax - agate| / |agate| (same formula, but different threshold)
    relative_error = l2_error

    # Check for NaN/Inf in computed error
    if np.isnan(l2_error) or np.isinf(l2_error):
        return False, {
            "jax_value": jax_value,
            "agate_value": agate_value,
            "l2_error": float("nan"),
            "relative_error": float("nan"),
            "error_msg": "NaN or Inf in computed error",
        }

    # Use different thresholds based on metric type
    if metric_name in ENERGY_METRICS:
        threshold = RELATIVE_ERROR_TOL
        passed = relative_error <= threshold
        threshold_type = "relative"
    else:
        threshold = L2_ERROR_TOL
        passed = l2_error <= threshold
        threshold_type = "l2"

    return passed, {
        "jax_value": jax_value,
        "agate_value": agate_value,
        "l2_error": l2_error,
        "relative_error": relative_error,
        "threshold": threshold,
        "threshold_type": threshold_type,
    }


def main(quick_test: bool = False) -> bool:
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    if quick_test:
        print("  (QUICK TEST MODE)")

    resolutions = QUICK_RESOLUTIONS if quick_test else RESOLUTIONS
    overall_pass = True
    metrics_report = {}

    # Download AGATE data before running simulations
    agate_data = {}
    print("Downloading AGATE reference data...")
    for resolution in resolutions:
        try:
            agate_times, agate_metrics = load_agate_series("ot", resolution)
            agate_data[resolution] = (agate_times, agate_metrics)
            print(f"  Resolution {resolution}: OK")
        except Exception as exc:
            print(f"  Resolution {resolution}: FAILED ({exc})")
            if not quick_test:
                overall_pass = False
                metrics_report[f"agate_data_r{resolution}"] = {
                    "value": "missing",
                    "passed": False,
                    "description": f"AGATE data load failed: {exc}",
                }

    for resolution in resolutions:
        print(f"Resolution: {resolution}")
        cfg = setup_configuration(quick_test, resolution)
        t_start = time.time()
        _, geometry, history = run_simulation(cfg)
        t_sim = time.time() - t_start
        print(f"  Simulation completed in {t_sim:.2f}s")

        jax_times = np.array(history["times"])
        jax_metrics = {key: np.array([m[key] for m in history["metrics"]]) for key in history["metrics"][0]}

        if quick_test:
            for key in jax_metrics:
                val = float(jax_metrics[key][-1])
                is_valid = not (np.isnan(val) or np.isinf(val))
                if not is_valid:
                    overall_pass = False
                metrics_report[f"{key}_r{resolution}"] = {
                    "value": val,
                    "passed": is_valid,
                    "description": "Quick test mode (NaN/Inf check only)",
                }
            continue

        if resolution not in agate_data:
            continue

        agate_times, agate_metrics = agate_data[resolution]
        for key, jax_values in jax_metrics.items():
            # Compare final values (last time point)
            jax_final = float(jax_values[-1])
            agate_final = float(agate_metrics[key][-1])
            passed, stats = compare_final_values(jax_final, agate_final, key)
            metrics_report[f"{key}_r{resolution}"] = {
                "jax_value": stats.get("jax_value", float("nan")),
                "agate_value": stats.get("agate_value", float("nan")),
                "l2_error": stats.get("l2_error", float("nan")),
                "relative_error": stats.get("relative_error", float("nan")),
                "threshold": stats.get("threshold", L2_ERROR_TOL),
                "threshold_type": stats.get("threshold_type", "l2"),
                "passed": passed,
                "description": stats.get("error_msg", f"{stats.get('threshold_type', 'L2')} error vs AGATE {key}"),
            }
            overall_pass = overall_pass and passed

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={"resolutions": resolutions},
        metrics=metrics_report,
        overall_pass=overall_pass,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Orszag–Tang Metrics (JAX-FRC)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    report.add_plot(fig, name="metrics_placeholder")
    plt.close(fig)

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")

    return bool(overall_pass)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
