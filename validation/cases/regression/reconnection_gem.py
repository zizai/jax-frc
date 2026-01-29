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
)
from validation.utils.plots import (
    create_scalar_comparison_plot,
    create_error_threshold_plot,
    create_field_comparison_plot,
)


NAME = "reconnection_gem"
DESCRIPTION = "GEM Hall reconnection regression vs AGATE reference data"

RESOLUTIONS = (512,)  # Only 512 available in AGATE reference data
QUICK_RESOLUTIONS = (512,)
# Relaxed thresholds due to different numerical schemes
L2_ERROR_TOL = 0.20  # 20% for field L2 errors
RELATIVE_ERROR_TOL = 0.20  # 20% for scalar metrics

NGAS = 4
NMAG = 3
IRHO = 0
IMX = 1
IMY = 2
IMZ = 3


def setup_configuration(quick_test: bool, resolution: int) -> dict:
    # AGATE reference data is at t=12.0
    # AGATE grid is 512 x 256 (nx x ny in AGATE's xy-plane)
    # JAX uses xz-plane, so we need nx=512, nz=256
    if quick_test:
        return {
            "nx": resolution,
            "nz": resolution // 2,  # Match AGATE aspect ratio
            "t_end": 0.1,  # Short run to verify stability
            "dt": 1e-4,
            "lambda_": 0.5,
            "psi1": 0.1,
            "B0": 1.0,
            "n0": 1.0,
            "n_b": 0.2,
        }
    return {
        "nx": resolution,
        "nz": resolution // 2,  # Match AGATE aspect ratio
        "t_end": 12.0,  # Match AGATE reference data time
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
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / case / str(resolution)

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
        print("  (QUICK TEST MODE)")
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
            loader.ensure_files("gem", resolution)
            print(f"  Resolution {resolution}: OK")
        except Exception as exc:
            print(f"  Resolution {resolution}: FAILED ({exc})")
    print()

    for resolution in resolutions:
        print(f"Resolution {resolution}: ", end="", flush=True)
        cfg = setup_configuration(quick_test, resolution)
        t_start = time.time()
        final_state, geometry, history = run_simulation(cfg)
        t_sim = time.time() - t_start
        print(f"[{t_sim:.2f}s]")

        # Load AGATE reference data
        # Note: AGATE only has t=12.0 data, so we always compare against final state
        try:
            agate_fields = load_agate_fields("gem", resolution, use_initial=False)
            agate_times, agate_scalar_metrics = load_agate_series("gem", resolution)
        except Exception as exc:
            print(f"  ERROR: Failed to load AGATE data: {exc}")
            overall_pass = False
            continue

        # In quick test mode, note the time mismatch
        if quick_test:
            print("  Quick test: simulation ran to t=0.1, comparing against AGATE t=12.0")
            print("  (Results will differ due to different simulation times)")
            print()

        # Compute field L2 errors
        field_errors = compute_field_l2_errors(final_state, agate_fields)
        print()  # Add blank line for readability
        print_field_l2_table(field_errors, L2_ERROR_TOL)
        print()

        # Compute scalar metrics comparison
        jax_final_metrics = compute_metrics(
            final_state.n, final_state.p, final_state.v, final_state.B,
            geometry.dx, geometry.dy, geometry.dz
        )

        # Use final AGATE scalar metrics (only t=12.0 data available)
        agate_idx = -1
        scalar_results = {}
        for key in jax_final_metrics:
            jax_val = jax_final_metrics[key]
            agate_val = float(agate_scalar_metrics[key][agate_idx])
            passed, stats = compare_final_values(jax_val, agate_val, key)
            scalar_results[key] = {
                'jax_value': jax_val,
                'agate_value': agate_val,
                'relative_error': stats.get('relative_error', 0),
                'threshold': stats.get('threshold', RELATIVE_ERROR_TOL),
                'passed': passed,
            }

        print_scalar_metrics_table(scalar_results)
        print()

        # Summary for this resolution
        field_passed = sum(1 for e in field_errors.values() if e <= L2_ERROR_TOL)
        scalar_passed = sum(1 for m in scalar_results.values() if m['passed'])
        total_checks = len(field_errors) + len(scalar_results)
        total_passed = field_passed + scalar_passed
        res_pass = total_passed == total_checks
        overall_pass = overall_pass and res_pass

        print(f"  Summary: {total_passed}/{total_checks} PASS")
        print()

        # Store results for report generation
        all_results[resolution] = {
            'field_errors': field_errors,
            'scalar_metrics': scalar_results,
            'jax_state': final_state,
            'agate_fields': agate_fields,
        }

        # Store metrics for report
        for field, error in field_errors.items():
            all_metrics[f"{field}_l2_r{resolution}"] = {
                'value': error,
                'threshold': L2_ERROR_TOL,
                'passed': error <= L2_ERROR_TOL,
                'description': f'{field} L2 error vs AGATE',
            }
        for key, data in scalar_results.items():
            all_metrics[f"{key}_r{resolution}"] = {
                'jax_value': data['jax_value'],
                'agate_value': data['agate_value'],
                'relative_error': data['relative_error'],
                'threshold': data['threshold'],
                'passed': data['passed'],
                'description': f'{key} relative error vs AGATE',
            }

    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={
            "resolutions": resolutions,
            "L2_threshold": L2_ERROR_TOL,
            "relative_threshold": RELATIVE_ERROR_TOL,
        },
        metrics=all_metrics,
        overall_pass=overall_pass,
    )

    # Generate plots for each resolution
    for resolution, data in all_results.items():
        # Plot 1: Scalar metrics comparison (bar chart)
        fig_scalar = create_scalar_comparison_plot(
            data['scalar_metrics'], resolution
        )
        report.add_plot(fig_scalar, name=f"scalar_comparison_r{resolution}",
                       caption=f"JAX vs AGATE scalar metrics at resolution {resolution}")
        plt.close(fig_scalar)

        # Plot 2: Error vs threshold summary
        fig_error = create_error_threshold_plot(
            data['field_errors'], data['scalar_metrics'],
            L2_ERROR_TOL, RELATIVE_ERROR_TOL
        )
        report.add_plot(fig_error, name=f"error_summary_r{resolution}",
                       caption=f"All errors as percentage of threshold at resolution {resolution}")
        plt.close(fig_error)

        # Plot 3: Field comparison contours (density)
        jax_density = np.asarray(data['jax_state'].n)[:, 0, :]
        agate_density = data['agate_fields']['density'][:, 0, :]
        fig_density = create_field_comparison_plot(
            jax_density, agate_density,
            'density', resolution, data['field_errors']['density']
        )
        report.add_plot(fig_density, name=f"density_comparison_r{resolution}",
                       caption=f"Density field comparison at resolution {resolution}")
        plt.close(fig_density)

        # Plot 4: Field comparison contours (magnetic field Bz)
        jax_bz = np.asarray(data['jax_state'].B)[:, 0, :, 2]
        agate_bz = data['agate_fields']['magnetic_field'][:, 0, :, 2]
        fig_bz = create_field_comparison_plot(
            jax_bz, agate_bz,
            'magnetic_field_Bz', resolution, data['field_errors']['magnetic_field']
        )
        report.add_plot(fig_bz, name=f"magnetic_field_comparison_r{resolution}",
                       caption=f"Magnetic field Bz comparison at resolution {resolution}")
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
