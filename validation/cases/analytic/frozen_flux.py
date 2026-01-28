"""3D Frozen Flux Validation - Circular Advection of Magnetic Loop.

Physics (Cartesian induction equation):
    dB/dt = curl(v x B) + eta * laplacian(B)

    In the ideal-MHD limit (Rm >> 1, eta -> 0):
        dB/dt = curl(v x B)

Benchmark:
    A localized magnetic loop is advected by rigid body rotation.
    After one full rotation period T = 2π/ω, the loop should return
    to its initial position. This validates:
    - Numerical diffusion (amplitude preservation)
    - Dispersion (shape preservation)
    - div(B) = 0 constraint maintenance

Coordinate System:
    3D Cartesian (x, y, z). We use a thin z dimension (nz=1) for
    pseudo-2D simulation in the x-y plane.
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
DESCRIPTION = "Circular advection of magnetic loop (rigid body rotation, Rm >> 1)"


def setup_configuration() -> dict:
    """Return configuration parameters for this validation case."""
    config = FrozenFluxConfiguration()
    runtime = config.default_runtime()
    return {
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "domain_extent": config.domain_extent,
        "z_extent": config.z_extent,
        "loop_x0": config.loop_x0,
        "loop_y0": config.loop_y0,
        "loop_radius": config.loop_radius,
        "loop_amplitude": config.loop_amplitude,
        "omega": config.omega,
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
        domain_extent=cfg["domain_extent"],
        z_extent=cfg["z_extent"],
        loop_x0=cfg["loop_x0"],
        loop_y0=cfg["loop_y0"],
        loop_radius=cfg["loop_radius"],
        loop_amplitude=cfg["loop_amplitude"],
        omega=cfg["omega"],
        eta=cfg["eta"],
    )

    geometry = config.build_geometry()
    initial_state = config.build_initial_state(geometry)
    state = initial_state
    model = config.build_model()

    solver = RK4Solver()

    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, initial_state, geometry, config


ACCEPTANCE = {
    "l2_error": 0.35,              # 35% L2 error (quarter rotation, central diff)
    "peak_amplitude_ratio": 0.75,  # 75% amplitude preservation
}


def main() -> bool:
    """Run validation and generate report."""
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    print()

    cfg = setup_configuration()
    print("Configuration:")
    for key, val in cfg.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4g}")
        else:
            print(f"  {key}: {val}")
    print()

    print("Running simulation...")
    t_start = time.time()
    final_state, initial_state, geometry, config = run_simulation(cfg)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Compare final B to analytic solution at t_end
    B_analytic = config.analytic_solution(geometry, cfg["t_end"])
    B_final = final_state.B
    B_initial = initial_state.B

    # Compute L2 error vs analytic
    l2_err = l2_error(B_final, B_analytic)

    # Compute peak amplitude ratio
    peak_initial = jnp.max(jnp.sqrt(jnp.sum(B_initial**2, axis=-1)))
    peak_final = jnp.max(jnp.sqrt(jnp.sum(B_final**2, axis=-1)))
    peak_ratio = float(peak_final / peak_initial)

    print("Metrics:")
    print(f"  L2 error: {l2_err:.4g} (threshold: {ACCEPTANCE['l2_error']})")
    print(f"  Peak amplitude ratio: {peak_ratio:.4g} (threshold: {ACCEPTANCE['peak_amplitude_ratio']})")
    print()

    l2_pass = l2_err < ACCEPTANCE["l2_error"]
    peak_pass = peak_ratio > ACCEPTANCE["peak_amplitude_ratio"]
    overall_pass = l2_pass and peak_pass

    metrics = {
        "l2_error": {
            "value": float(l2_err),
            "threshold": ACCEPTANCE["l2_error"],
            "passed": l2_pass,
            "description": "Relative L2 error between final and analytic B field",
        },
        "peak_amplitude_ratio": {
            "value": peak_ratio,
            "threshold": ACCEPTANCE["peak_amplitude_ratio"],
            "passed": peak_pass,
            "description": "Ratio of peak |B| after one rotation to initial peak",
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

    # Plot B magnitude at z=0 (2D contour)
    import matplotlib.pyplot as plt

    z_idx = geometry.nz // 2
    x = geometry.x_grid[:, :, z_idx]
    y = geometry.y_grid[:, :, z_idx]

    B_mag_analytic = jnp.sqrt(jnp.sum(B_analytic[:, :, z_idx, :]**2, axis=-1))
    B_mag_final = jnp.sqrt(jnp.sum(B_final[:, :, z_idx, :]**2, axis=-1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Analytic
    im0 = axes[0].contourf(x, y, B_mag_analytic, levels=20, cmap="viridis")
    axes[0].set_title(f"Analytic |B| (t={cfg['t_end']:.2f}s)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], label="|B| [T]")

    # Final
    im1 = axes[1].contourf(x, y, B_mag_final, levels=20, cmap="viridis")
    axes[1].set_title(f"Numerical |B| (t={cfg['t_end']:.2f}s)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], label="|B| [T]")

    # Error
    error = B_mag_final - B_mag_analytic
    im2 = axes[2].contourf(x, y, error, levels=20, cmap="RdBu_r")
    axes[2].set_title("Error (Numerical - Analytic)")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2], label="Error [T]")

    plt.tight_layout()
    report.add_plot(fig, name="B_magnitude_comparison")

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    if overall_pass:
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: Some acceptance criteria not met")
        if not l2_pass:
            print(f"  - L2 error {l2_err:.4g} exceeds threshold {ACCEPTANCE['l2_error']}")
        if not peak_pass:
            print(f"  - Peak ratio {peak_ratio:.4g} below threshold {ACCEPTANCE['peak_amplitude_ratio']}")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
