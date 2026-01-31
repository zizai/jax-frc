"""Brio-Wu MHD Shock Tube Validation

Physics:
    The Brio-Wu shock tube is a standard test problem for MHD codes that
    validates shock-capturing numerics. It produces a complex wave structure
    including fast and slow magnetosonic shocks, compound waves, and contact
    discontinuities.

    Initial conditions (z < 0.5 | z > 0.5) on domain [0, 1]:
        rho:  1.0    | 0.125
        p:    1.0    | 0.1
        By:   1.0    | -1.0  (tangential component across z)
        Bz:   0.75   | 0.75 (constant normal component)

    The discontinuity at z=0.5 evolves into:
        - Fast rarefaction wave (left-going)
        - Compound wave (slow shock + rotational discontinuity)
        - Contact discontinuity
        - Slow shock (right-going)
        - Fast shock (right-going)

    This test validates that the numerical scheme correctly captures
    the MHD wave structure and maintains energy conservation.

Numerics:
    JAX-FRC matches the AGATE baseline configuration for direct comparison:
    ideal MHD finite volume (HLL), MC-beta limiter (beta=1.3), CFL=0.4,
    RK2 time integration, and t_end=0.2.

Notes:
    This case compares JAX-FRC against AGATE reference data using
    L2 field errors and relative errors for aggregate metrics. Snapshot
    times are loaded from the AGATE config when available.

Reference:
    Brio & Wu (1988), "An upwind differencing scheme for the equations
    of ideal magnetohydrodynamics", Journal of Computational Physics, 75(2).
"""

import re
import time
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.brio_wu import BrioWuConfiguration
from jax_frc.models.finite_volume_mhd import FiniteVolumeMHD
from jax_frc.solvers import RK2Solver
from jax_frc.validation.metrics import l2_error
from validation.utils.agate_data import AgateDataLoader
from validation.utils.reporting import (
    ValidationReport,
    print_field_l2_table,
    print_aggregate_metrics_table,
)


# Case metadata
NAME = "brio_wu"
DESCRIPTION = "Brio-Wu MHD shock tube"

L2_ERROR_TOL = 0.20  # 20% field L2 threshold
RELATIVE_ERROR_TOL = 0.20  # 20% scalar/aggregate threshold
AGGREGATE_KEYS = [
    "kinetic_fraction",
    "magnetic_fraction",
    "mean_energy_density",
]


def setup_configuration(quick_test: bool = False) -> dict:
    """Return configuration parameters for this validation case.

    Returns:
        Dictionary with all physics and numerical parameters.
    """
    cfg = {
        "nz": 512,            # Grid points in z (high resolution for shocks)
        "gamma": 2.0,         # Adiabatic index (Brio-Wu standard)
        "t_end": 0.2,         # End time [Alfven units] (AGATE baseline)
    }
    if quick_test:
        cfg = {**cfg, "nz": 256}
    return cfg


def load_agate_config(resolution: list[int]) -> dict:
    """Load AGATE config including snapshot_times."""
    loader = AgateDataLoader()
    loader.ensure_files("bw", resolution)
    case_dir = Path(loader.cache_dir) / "brio_wu" / str(resolution[0])
    config_path = case_dir / f"brio_wu_{resolution[0]}.config.yaml"
    if not config_path.exists():
        return {"snapshot_times": None}
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_state_vector(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = vec[0]
    mom = np.stack([vec[1], vec[2], vec[3]], axis=-1)
    v = np.divide(mom, rho[..., None], out=np.zeros_like(mom), where=rho[..., None] != 0)
    B = np.moveaxis(vec[4:7], 0, -1)
    p_idx = 7
    p = vec[p_idx] if vec.shape[0] > p_idx else np.zeros_like(rho)
    return rho, p, v, B


def _strip_ghosts(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        slices = [
            slice(2, -2) if dim > 4 else slice(0, dim)
            for dim in arr.shape
        ]
        return arr[tuple(slices)]
    if arr.ndim == 4:
        slices = [
            slice(2, -2) if dim > 4 else slice(0, dim)
            for dim in arr.shape[:3]
        ]
        return arr[tuple(slices + [slice(None)])]
    return arr


def load_agate_snapshot(resolution: list[int], snapshot_idx: int) -> dict:
    """Load spatial fields from a specific AGATE snapshot."""
    loader = AgateDataLoader()
    loader.ensure_files("bw", resolution)
    case_dir = Path(loader.cache_dir) / "brio_wu" / str(resolution[0])
    state_path = case_dir / f"brio_wu_{resolution[0]}.state_{snapshot_idx:06d}.h5"

    if not state_path.exists():
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
    rho = _strip_ghosts(rho)
    p = _strip_ghosts(p)
    v = _strip_ghosts(v)
    B = _strip_ghosts(B)

    # Map AGATE x-direction to JAX z-direction
    rho = np.transpose(rho, (1, 2, 0))
    p = np.transpose(p, (1, 2, 0))
    v = np.transpose(v, (1, 2, 0, 3))
    B = np.transpose(B, (1, 2, 0, 3))

    # Rotate vector components: AGATE x -> JAX z
    v = v[..., [2, 1, 0]]
    B = B[..., [2, 1, 0]]

    mom = rho[..., None] * v
    return {"density": rho, "momentum": mom, "magnetic_field": B, "pressure": p}


def compute_metrics(rho, p, v, B, dx: float, dy: float, dz: float) -> dict:
    """Compute normalized scalar metrics for comparison."""
    rho = np.asarray(rho)
    p = np.asarray(p)
    v = np.asarray(v)
    B = np.asarray(B)
    v_sq = np.sum(v**2, axis=-1)
    B_sq = np.sum(B**2, axis=-1)

    kinetic = 0.5 * rho * v_sq
    magnetic = 0.5 * B_sq
    gamma = 2.0
    thermal = p / (gamma - 1.0)

    total_density = float(np.mean(kinetic + magnetic + thermal))
    kinetic_density = float(np.mean(kinetic))
    magnetic_density = float(np.mean(magnetic))

    kinetic_fraction = kinetic_density / total_density if total_density > 0 else 0.0
    magnetic_fraction = magnetic_density / total_density if total_density > 0 else 0.0

    return {
        "kinetic_fraction": kinetic_fraction,
        "magnetic_fraction": magnetic_fraction,
        "mean_energy_density": total_density,
    }


def compute_field_metrics(jax_field: np.ndarray, agate_field: np.ndarray) -> dict:
    """Compute per-field metrics: L2 error, max abs error, relative error."""
    jax_field = np.asarray(jax_field)
    agate_field = np.asarray(agate_field)
    diff = jax_field - agate_field
    l2 = float(l2_error(jax_field, agate_field))
    max_abs = float(np.max(np.abs(diff)))
    denom = np.max(np.abs(agate_field))
    rel = float(np.sqrt(np.mean(diff**2)) / denom) if denom > 1e-10 else 0.0
    return {"l2_error": l2, "max_abs_error": max_abs, "relative_error": rel}


def validate_all_snapshots(
    jax_states: list,
    resolution: list[int],
    snapshot_times: list[float],
    agate_snapshot_times: list[float] | None = None,
) -> list[dict]:
    """Compare JAX-FRC vs AGATE at each snapshot."""
    all_errors = []
    agate_times = np.array(agate_snapshot_times) if agate_snapshot_times is not None else None
    for i, (jax_state, t) in enumerate(zip(jax_states, snapshot_times)):
        if agate_times is not None:
            snapshot_idx = int(np.argmin(np.abs(agate_times - t)))
        else:
            snapshot_idx = i
        agate_fields = load_agate_snapshot(resolution, snapshot_idx=snapshot_idx)

        field_errors = {}
        field_errors["density"] = compute_field_metrics(np.asarray(jax_state.n), agate_fields["density"])
        jax_mom = np.asarray(jax_state.n)[..., None] * np.asarray(jax_state.v)
        field_errors["momentum"] = compute_field_metrics(jax_mom, agate_fields["momentum"])
        field_errors["magnetic_field"] = compute_field_metrics(np.asarray(jax_state.B), agate_fields["magnetic_field"])
        field_errors["pressure"] = compute_field_metrics(np.asarray(jax_state.p), agate_fields["pressure"])
        all_errors.append({"time": t, "errors": field_errors})
    return all_errors


def compute_aggregate_metrics(jax_history: dict, agate_times: np.ndarray, agate_metrics: dict) -> dict:
    """Compute time-series comparison metrics for aggregate quantities."""
    results = {}
    jax_times = np.array(jax_history["times"])
    jax_metrics_by_key = {}
    for key in AGGREGATE_KEYS:
        jax_metrics_by_key[key] = np.array([m[key] for m in jax_history["metrics"]])

    for key in jax_metrics_by_key:
        jax_vals = jax_metrics_by_key[key]
        agate_vals = agate_metrics[key]
        if len(jax_vals) != len(agate_vals):
            jax_vals = np.interp(agate_times, jax_times, jax_vals)
        residuals = jax_vals - agate_vals
        mean_resid = float(np.mean(residuals))
        std_resid = float(np.std(residuals))
        denom = max(np.mean(np.abs(agate_vals)), 1e-10)
        rel_error = float(np.mean(np.abs(residuals)) / denom)
        results[key] = {
            "mean_residual": mean_resid,
            "std_residual": std_resid,
            "relative_error": rel_error,
        }
    return results


def load_agate_series(resolution: list[int]) -> tuple[np.ndarray, dict]:
    loader = AgateDataLoader()
    loader.ensure_files("bw", resolution)
    case_dir = Path(loader.cache_dir) / "brio_wu" / str(resolution[0])
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
        dy = float(sub.attrs.get("dy", 1.0))
        dz = float(sub.attrs.get("dz", 1.0))
        if dz == 0.0:
            dz = 1.0

    times = []
    metrics = {key: [] for key in AGGREGATE_KEYS}
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
            rho = _strip_ghosts(rho)
            p = _strip_ghosts(p)
            v = _strip_ghosts(v)
            B = _strip_ghosts(B)
            rho = np.transpose(rho, (1, 2, 0))
            p = np.transpose(p, (1, 2, 0))
            v = np.transpose(v, (1, 2, 0, 3))
            B = np.transpose(B, (1, 2, 0, 3))
            v = v[..., [2, 1, 0]]
            B = B[..., [2, 1, 0]]
            metric = compute_metrics(rho, p, v, B, dx, dy, dz)
            times.append(float(time_attr))
            for key in metrics:
                metrics[key].append(metric[key])

    return np.array(times), {k: np.array(v) for k, v in metrics.items()}


def run_simulation_with_snapshots(cfg: dict, snapshot_times: list[float]) -> tuple[list, any, dict]:
    """Run JAX-FRC, capturing state at each snapshot time."""
    config = BrioWuConfiguration(
        nz=cfg["nz"],
        gamma=cfg["gamma"],
    )
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = FiniteVolumeMHD(
        gamma=cfg["gamma"],
        riemann_solver="hll",
        limiter_beta=1.3,
        cfl=0.4,
        use_divergence_cleaning=False,
    )
    solver = RK2Solver()

    states = []
    history = {"times": [], "metrics": []}

    for target_time in snapshot_times:
        while state.time < target_time - 1e-12:
            step_dt = min(model.compute_stable_dt(state, geometry), target_time - state.time)
            state = solver.step_checked(state, step_dt, model, geometry)
        states.append(state)
        metrics = compute_metrics(
            state.n, state.p, state.v, state.B, geometry.dx, geometry.dy, geometry.dz
        )
        history["times"].append(float(state.time))
        history["metrics"].append(metrics)

    return states, geometry, history



def main(quick_test: bool = False) -> bool:
    """Run validation and generate report.

    Returns:
        True if all acceptance criteria pass, False otherwise.
    """
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    if quick_test:
        print("  (QUICK TEST MODE)")
    print()

    # Setup
    cfg = setup_configuration(quick_test=quick_test)
    resolution = [cfg["nz"], 1, 1]
    print("Configuration:")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print(f"  L2 threshold: {L2_ERROR_TOL}")
    print(f"  Relative threshold: {RELATIVE_ERROR_TOL}")
    print()

    print("Preparing AGATE reference data...")
    loader = AgateDataLoader()
    loader.ensure_files("bw", resolution)
    agate_config = load_agate_config(resolution)

    snapshot_times = agate_config.get("snapshot_times")
    if snapshot_times is None:
        snapshot_times = [0.0, cfg["t_end"]]
    print(f"  Using {len(snapshot_times)} snapshots (matching AGATE)")
    print()

    print(f"Running simulation to t={cfg['t_end']}...", end="", flush=True)
    t_start = time.time()
    jax_states, geometry, history = run_simulation_with_snapshots(cfg, snapshot_times)
    t_sim = time.time() - t_start
    print(f" [{t_sim:.2f}s]")
    print()

    print(f"Validating {len(snapshot_times)} snapshots...")
    snapshot_errors = validate_all_snapshots(jax_states, resolution, snapshot_times, snapshot_times)
    final_field_errors = snapshot_errors[-1]["errors"]
    print()
    print_field_l2_table(final_field_errors, L2_ERROR_TOL)
    print()

    agate_times, agate_scalar_metrics = load_agate_series(resolution)
    aggregate_metrics = compute_aggregate_metrics(history, agate_times, agate_scalar_metrics)
    print_aggregate_metrics_table(aggregate_metrics, RELATIVE_ERROR_TOL)
    print()

    field_passed = sum(
        1 for stats in final_field_errors.values() if stats["l2_error"] <= L2_ERROR_TOL
    )
    agg_passed = sum(
        1 for stats in aggregate_metrics.values()
        if stats["relative_error"] <= RELATIVE_ERROR_TOL
    ) if aggregate_metrics else 0
    total_checks = len(final_field_errors) + len(aggregate_metrics)
    total_passed = field_passed + agg_passed
    overall_pass = total_passed == total_checks

    print(f"Summary: {total_passed}/{total_checks} PASS")
    print()

    metrics = {}
    for field, stats in final_field_errors.items():
        metrics[f"{field}_l2"] = {
            "value": stats["l2_error"],
            "threshold": L2_ERROR_TOL,
            "passed": stats["l2_error"] <= L2_ERROR_TOL,
            "description": f"{field} L2 error vs AGATE (final snapshot)",
        }
    for key, stats in aggregate_metrics.items():
        metrics[key] = {
            "value": stats["relative_error"],
            "threshold": RELATIVE_ERROR_TOL,
            "passed": stats["relative_error"] <= RELATIVE_ERROR_TOL,
            "description": f"{key} time-series relative error vs AGATE",
        }

    report_config = {
        **cfg,
        "resolution": resolution,
        "L2_threshold": L2_ERROR_TOL,
        "relative_threshold": RELATIVE_ERROR_TOL,
        "quick_test": quick_test,
    }
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=report_config,
        metrics=metrics,
        overall_pass=overall_pass,
        timing={"simulation": t_sim},
    )

    # Plot comparisons at final snapshot
    agate_fields = load_agate_snapshot(resolution, len(snapshot_errors) - 1)
    z = geometry.z_grid[0, 0, :]
    jax_density = np.asarray(jax_states[-1].n)[0, 0, :]
    agate_density = agate_fields["density"][0, 0, :]
    fig_density, ax_density = plt.subplots(figsize=(10, 6))
    ax_density.plot(z, jax_density, "b-", linewidth=1.5, label="JAX")
    ax_density.plot(z, agate_density, "r--", linewidth=1.5, label="AGATE")
    ax_density.set_xlabel("z")
    ax_density.set_ylabel("Density")
    ax_density.set_title(f"Density Profile at t={cfg['t_end']}")
    ax_density.legend(loc="upper right")
    ax_density.grid(True, alpha=0.3)
    fig_density.tight_layout()
    report.add_plot(fig_density, name="density_profile")

    jax_Bz = np.asarray(jax_states[-1].B)[0, 0, :, 2]
    agate_Bz = agate_fields["magnetic_field"][0, 0, :, 2]
    fig_Bz, ax_Bz = plt.subplots(figsize=(10, 6))
    ax_Bz.plot(z, jax_Bz, "b-", linewidth=1.5, label="JAX")
    ax_Bz.plot(z, agate_Bz, "r--", linewidth=1.5, label="AGATE")
    ax_Bz.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax_Bz.set_xlabel("z")
    ax_Bz.set_ylabel("Bz")
    ax_Bz.set_title(f"Normal Magnetic Field Profile at t={cfg['t_end']}")
    ax_Bz.legend(loc="upper right")
    ax_Bz.grid(True, alpha=0.3)
    fig_Bz.tight_layout()
    report.add_plot(fig_Bz, name="Bz_profile")

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()
    print("OVERALL: PASS" if overall_pass else "OVERALL: FAIL")
    return overall_pass


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
