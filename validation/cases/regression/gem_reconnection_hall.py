"""GEM Hall Reconnection Regression Against AGATE Reference Data.

Physics:
    Hall MHD GEM reconnection in a thin-y slab (xz plane).

Notes:
    This case enables the Hall term in both JAX-FRC and AGATE. It uses a
    shorter time horizon (t_end=2.0) to keep the Hall run tractable while
    preserving the quadrupole signature development.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.gem_reconnection import GEMReconnectionConfiguration
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers import Solver
from validation.cases.regression import gem_reconnection as base
from validation.utils.agate_data import AgateDataLoader
from validation.utils.agate_runner import run_agate_simulation
from validation.utils.reporting import (
    ValidationReport,
    print_field_l2_table,
    print_aggregate_metrics_table,
)
from validation.utils.plots import (
    create_field_comparison_plot,
    create_timeseries_comparison_plot,
    create_field_error_evolution_plot,
)


NAME = "gem_reconnection_hall"
DESCRIPTION = "GEM reconnection regression vs AGATE Hall-MHD reference data"

RESOLUTIONS = ([64, 32, 1],)

L2_ERROR_TOL = base.L2_ERROR_TOL
RELATIVE_ERROR_TOL = base.RELATIVE_ERROR_TOL


def setup_configuration(resolution: list[int]) -> dict:
    cfg = base.setup_configuration(resolution)
    cfg["t_end"] = 2.0
    cfg["dt"] = 1e-3
    cfg["use_cfl"] = True
    return cfg


def run_simulation_with_snapshots(
    cfg: dict,
    snapshot_times: list[float],
) -> tuple[list, object, dict]:
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

    model = ExtendedMHD(
        eta=config.eta,
        include_hall=True,
        include_electron_pressure=False,
        apply_divergence_cleaning=True,
        normalized_units=True,
    )
    solver = Solver.create({"type": "rk4"})

    dt_cfl = float(model.compute_stable_dt(state, geometry))
    dt = min(dt_cfl, float(cfg.get("dt", dt_cfl)))

    states: list = []
    history = {"times": [], "metrics": []}

    for target_time in snapshot_times:
        while state.time < target_time - 1e-12:
            step_dt = min(dt, target_time - state.time)
            state = solver.step_checked(state, step_dt, model, geometry)

        states.append(state)
        metrics = base.compute_metrics(
            state.n, state.p, state.v, state.B, geometry.dx, geometry.dy, geometry.dz
        )
        history["times"].append(float(state.time))
        history["metrics"].append(metrics)

    return states, geometry, history


def main() -> bool:
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    print()

    print("Configuration:")
    resolutions = RESOLUTIONS
    print(f"  resolutions: {resolutions}")
    print(f"  Field L2 threshold: {L2_ERROR_TOL} ({L2_ERROR_TOL*100:.0f}%)")
    print(f"  Relative error threshold: {RELATIVE_ERROR_TOL} ({RELATIVE_ERROR_TOL*100:.0f}%)")
    print()

    overall_pass = True
    all_results = {}
    all_metrics = {}
    case_name = "gem_reconnection_hall"

    print("Preparing AGATE reference data...")
    for resolution in resolutions:
        try:
            loader = AgateDataLoader()
            agate_start = time.time()
            loader.ensure_files(case_name, resolution)
            agate_elapsed = time.time() - agate_start
            print(f"  Resolution {resolution[0]}: OK ({agate_elapsed:.2f}s)")
        except Exception as exc:
            print(f"  Resolution {resolution[0]}: FAILED ({exc})")
    print()

    for resolution in resolutions:
        res_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
        print(f"Resolution {res_str}:")

        try:
            agate_config = base.load_agate_config(case_name, resolution)
        except Exception as exc:
            print(f"  ERROR: Failed to load AGATE config: {exc}")
            overall_pass = False
            continue

        if agate_config.get("resolution") != list(resolution):
            try:
                print(
                    "  AGATE cache resolution mismatch; regenerating "
                    f"{resolution[0]}x{resolution[1]}x{resolution[2]}..."
                )
                loader = AgateDataLoader()
                output_dir = Path(loader.cache_dir) / case_name / str(resolution[0])
                run_agate_simulation(case_name, list(resolution), output_dir, overwrite=True)
                agate_config = base.load_agate_config(case_name, resolution)
            except Exception as exc:
                print(f"  ERROR: Failed to regenerate AGATE data: {exc}")
                overall_pass = False
                continue

        agate_snapshot_times = agate_config.get("snapshot_times")
        if agate_snapshot_times is None:
            agate_snapshot_times = [0.0, 2.0]
        snapshot_times = list(agate_snapshot_times)
        print(f"  Using {len(snapshot_times)} snapshots (matching AGATE)")

        cfg = setup_configuration(resolution)
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

        print(f"  Validating {len(snapshot_times)} snapshots...")
        try:
            snapshot_errors = base.validate_all_snapshots(
                jax_states, case_name, resolution, snapshot_times, agate_snapshot_times
            )
        except Exception as exc:
            print(f"  ERROR: Failed to validate snapshots: {exc}")
            overall_pass = False
            continue

        try:
            agate_times, agate_scalar_metrics = base.load_agate_series(case_name, resolution)
            aggregate_metrics = base.compute_aggregate_metrics(
                history, agate_times, agate_scalar_metrics
            )
        except Exception as exc:
            print(f"  WARNING: Failed to compute aggregate metrics: {exc}")
            aggregate_metrics = {}
            agate_times = None
            agate_scalar_metrics = None

        final_field_errors = snapshot_errors[-1]["errors"]
        print()
        print_field_l2_table(final_field_errors, L2_ERROR_TOL)
        print()

        if aggregate_metrics:
            print_aggregate_metrics_table(aggregate_metrics, RELATIVE_ERROR_TOL)
            print()

        field_passed = sum(
            1 for stats in final_field_errors.values()
            if stats["l2_error"] <= L2_ERROR_TOL
        )
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

        all_results[resolution[0]] = {
            "field_errors": final_field_errors,
            "snapshot_errors": snapshot_errors,
            "aggregate_metrics": aggregate_metrics,
            "jax_states": jax_states,
            "history": history,
            "agate_times": agate_times,
            "agate_scalar_metrics": agate_scalar_metrics,
            "resolution": resolution,
        }

        for field, stats in final_field_errors.items():
            all_metrics[f"{field}_l2_r{resolution[0]}"] = {
                "value": stats["l2_error"],
                "threshold": L2_ERROR_TOL,
                "passed": stats["l2_error"] <= L2_ERROR_TOL,
                "description": f"{field} L2 error vs AGATE (final snapshot)",
            }
        for key, stats in aggregate_metrics.items():
            all_metrics[f"{key}_r{resolution[0]}"] = {
                "value": stats["relative_error"],
                "threshold": RELATIVE_ERROR_TOL,
                "passed": stats["relative_error"] <= RELATIVE_ERROR_TOL,
                "description": f"{key} time-series relative error vs AGATE",
            }

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={
            "resolutions": [list(r) for r in resolutions],
            "L2_threshold": L2_ERROR_TOL,
            "relative_threshold": RELATIVE_ERROR_TOL,
            "quick_test": False,
        },
        metrics=all_metrics,
        overall_pass=overall_pass,
    )

    for res_key, data in all_results.items():
        resolution = data["resolution"]

        fig_error_evol = create_field_error_evolution_plot(
            data["snapshot_errors"], L2_ERROR_TOL, resolution
        )
        report.add_plot(
            fig_error_evol,
            name=f"field_error_evolution_r{res_key}",
            caption=f"Per-field L2 error evolution at resolution {res_key}",
        )
        plt.close(fig_error_evol)

        jax_times = np.array(data["history"]["times"])
        agate_times = data["agate_times"]
        for metric_name in ["kinetic_fraction", "magnetic_fraction"]:
            jax_vals = np.array([m[metric_name] for m in data["history"]["metrics"]])
            agate_vals = data["agate_scalar_metrics"][metric_name]
            fig_ts = create_timeseries_comparison_plot(
                jax_times, jax_vals, agate_times, agate_vals,
                metric_name, resolution
            )
            report.add_plot(
                fig_ts,
                name=f"{metric_name}_timeseries_r{res_key}",
                caption=f"{metric_name} time evolution at resolution {res_key}",
            )
            plt.close(fig_ts)

        final_state = data["jax_states"][-1]
        try:
            agate_fields = base.load_agate_snapshot(
                case_name, resolution, len(data["snapshot_errors"]) - 1
            )
        except Exception:
            agate_fields = base.load_agate_fields(case_name, resolution, use_initial=False)

        jax_density = np.asarray(final_state.n)[:, 0, :]
        agate_density = agate_fields["density"][:, 0, :]
        fig_density = create_field_comparison_plot(
            jax_density, agate_density,
            "density", res_key, data["field_errors"]["density"]["l2_error"],
        )
        report.add_plot(
            fig_density,
            name=f"density_comparison_r{res_key}",
            caption=f"Density field comparison at resolution {res_key}",
        )
        plt.close(fig_density)

        jax_bz = np.asarray(final_state.B)[:, 0, :, 2]
        agate_bz = agate_fields["magnetic_field"][:, 0, :, 2]
        fig_bz = create_field_comparison_plot(
            jax_bz, agate_bz,
            "magnetic_field_Bz", res_key,
            data["field_errors"]["magnetic_field"]["l2_error"],
        )
        report.add_plot(
            fig_bz,
            name=f"magnetic_field_comparison_r{res_key}",
            caption=f"Magnetic field Bz comparison at resolution {res_key}",
        )
        plt.close(fig_bz)

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    if overall_pass:
        print("OVERALL: PASS (all resolutions passed)")
    else:
        print("OVERALL: FAIL (some checks failed)")

    return bool(overall_pass)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
