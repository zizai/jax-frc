"""3D Frozen Flux Validation (Cartesian Geometry).

Physics (Cartesian induction equation):
    ∂B/∂t = ∇×(v×B) + (1/μ0σ)∇²B

    In the ideal-MHD limit (Rm >> 1, η ≈ 1/μ0σ → 0):
        ∂B/∂t = ∇×(v×B)

    For uniform v and uniform B in Cartesian coordinates, ∇×(v×B) = 0,
    so B remains constant in time. This validates that the solver preserves
    a frozen-in uniform field under uniform advection.

Coordinate System:
    3D Cartesian (x, y, z). We use a thin y dimension to keep the case
    inexpensive while remaining fully 3D.
"""

import time
import sys
from pathlib import Path

import jax.numpy as jnp

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.frozen_flux import FrozenFluxConfiguration
from jax_frc.solvers.explicit import RK4Solver
from jax_frc.validation.metrics import l2_error
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error


NAME = "frozen_flux"
DESCRIPTION = "3D frozen flux validation in Cartesian geometry (uniform B_y remains constant)"


def setup_configuration() -> dict:
    """Return configuration parameters for this validation case."""
    config = FrozenFluxConfiguration()
    runtime = config.default_runtime()
    return {
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "r_min": config.r_min,
        "r_max": config.r_max,
        "z_extent": config.z_extent,
        "B_phi_0": config.B_phi_0,
        "v_r": config.v_r,
        "eta": config.eta,
        "t_end": runtime["t_end"],
        "dt": runtime["dt"],
    }


def run_simulation(cfg: dict) -> tuple:
    """Run the frozen flux simulation."""
    config = FrozenFluxConfiguration(
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        r_min=cfg["r_min"],
        r_max=cfg["r_max"],
        z_extent=cfg["z_extent"],
        B_phi_0=cfg["B_phi_0"],
        v_r=cfg["v_r"],
        eta=cfg["eta"],
    )

    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    solver = RK4Solver()

    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, geometry, config


ACCEPTANCE = {
    "l2_error": 0.01,
}


def main() -> bool:
    """Run validation and generate report."""
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    print()

    cfg = setup_configuration()
    print("Configuration:")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print()

    print("Running simulation...")
    t_start = time.time()
    final_state, geometry, config = run_simulation(cfg)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Expected B_y is uniform and constant
    B_y = final_state.B[..., 1]
    B_expected = jnp.ones_like(B_y) * cfg["B_phi_0"]

    l2_err = l2_error(B_y, B_expected)

    print("Metrics:")
    print(f"  L2 error: {l2_err:.4g} (threshold: {ACCEPTANCE['l2_error']})")
    print()

    l2_pass = l2_err < ACCEPTANCE["l2_error"]
    overall_pass = l2_pass

    metrics = {
        "l2_error": {
            "value": float(l2_err),
            "threshold": ACCEPTANCE["l2_error"],
            "passed": l2_pass,
            "description": "Relative L2 error vs uniform analytic solution",
        },
    }

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=metrics,
        overall_pass=overall_pass,
        timing={"simulation": t_sim},
    )

    # Plot along x at center y,z
    y_idx = geometry.ny // 2
    z_idx = geometry.nz // 2
    x = geometry.x_grid[:, y_idx, z_idx]
    B_y_line = B_y[:, y_idx, z_idx]
    B_expected_line = B_expected[:, y_idx, z_idx]

    fig_comp = plot_comparison(
        x=x,
        actual=B_y_line,
        expected=B_expected_line,
        labels=("Simulation", "Analytic"),
        title=f"B_y at y=0, z=0, t={cfg['t_end']:.2e}s",
        xlabel="x [m]",
        ylabel="B_y [T]",
    )
    report.add_plot(fig_comp, name="By_comparison")

    fig_err = plot_error(
        x=x,
        actual=B_y_line,
        expected=B_expected_line,
        title="B_y Error (Simulation - Analytic)",
        xlabel="x [m]",
        ylabel="Error [T]",
    )
    report.add_plot(fig_err, name="By_error")

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    if overall_pass:
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: L2 error exceeds threshold")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
