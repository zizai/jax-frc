"""GEM Reconnection Regression Against AGATE Reference Data.

Physics:
    Hall MHD GEM reconnection in a thin-y slab (cylindrical-style geometry).
    This case compares scalar time-series metrics against AGATE reference data
    via block-bootstrap mean relative error with 95% confidence intervals.
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
from validation.utils.regression import block_bootstrap_ci
from validation.utils.reporting import ValidationReport


NAME = "reconnection_gem"
DESCRIPTION = "GEM Hall reconnection regression vs AGATE reference data"

RESOLUTIONS = (256, 512, 1024)
QUICK_RESOLUTIONS = (256,)
ERROR_TOL = 0.2

NGAS = 4
NMAG = 3
IRHO = 0
IMX = 1
IMY = 2
IMZ = 3


def setup_configuration(quick_test: bool, resolution: int) -> dict:
    if quick_test:
        return {
            "nx": resolution,
            "nz": resolution * 2,
            "t_end": 0.1,
            "dt": 1e-3,
            "lambda_": 0.5,
            "psi1": 0.01,
            "B0": 1.0,
            "n0": 1.0,
            "n_b": 0.2,
        }
    return {
        "nx": resolution,
        "nz": resolution * 2,
        "t_end": 25.0,
        "dt": 0.01,
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
    ny = vec.shape[1]

    if ny > 1:
        dvz_dy = np.gradient(vz, dy, axis=1)
        dvx_dy = np.gradient(vx, dy, axis=1)
    else:
        dvz_dy = np.zeros_like(vz)
        dvx_dy = np.zeros_like(vx)

    dvy_dz = np.gradient(vy, dz, axis=2)
    dvx_dz = np.gradient(vx, dz, axis=2)
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
    model = config.build_model()
    solver = Solver.create({"type": "semi_implicit"})

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
        state = solver.step(state, dt, model, geometry)
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
    v = np.stack([vec[IMX], vec[IMY], vec[IMZ]], axis=-1)
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
        dx = float(f.attrs.get("dx", np.diff(f["x"][:]).mean()))
        dy = float(f.attrs.get("dy", np.diff(f["y"][:]).mean()))
        dz = float(f.attrs.get("dz", np.diff(f["z"][:]).mean()))

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


def compare_series(jax_times, jax_values, agate_times, agate_values) -> tuple[bool, dict]:
    if len(agate_times) < 2:
        agate_interp = np.full_like(jax_values, agate_values[0], dtype=float)
    else:
        agate_interp = np.interp(jax_times, agate_times, agate_values)
    denom = np.maximum(np.abs(agate_interp), 1e-8)
    errors = np.abs(jax_values - agate_interp) / denom
    mean_err, lo, hi = block_bootstrap_ci(errors, block_size=10, n_boot=300)
    passed = hi <= ERROR_TOL
    return passed, {"mean_error": mean_err, "ci_low": lo, "ci_high": hi}


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
            agate_times, agate_metrics = load_agate_series("gem", resolution)
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
                metrics_report[f"{key}_r{resolution}"] = {
                    "value": float(jax_metrics[key][-1]),
                    "passed": True,
                    "description": "Quick test mode (no regression)",
                }
            continue

        if resolution not in agate_data:
            continue

        agate_times, agate_metrics = agate_data[resolution]
        for key, jax_values in jax_metrics.items():
            passed, stats = compare_series(
                jax_times, jax_values, agate_times, agate_metrics[key]
            )
            metrics_report[f"{key}_r{resolution}"] = {
                "value": stats["mean_error"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "threshold": ERROR_TOL,
                "passed": passed,
                "description": f"Mean relative error (95% CI) vs AGATE {key}",
            }
            overall_pass = overall_pass and passed

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={"resolutions": resolutions},
        metrics=metrics_report,
        overall_pass=overall_pass or quick_test,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("GEM Metrics (JAX-FRC)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    report.add_plot(fig, name="metrics_placeholder")
    plt.close(fig)

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")

    return bool(overall_pass or quick_test)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
