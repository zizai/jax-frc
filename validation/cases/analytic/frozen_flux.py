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
import numpy as np

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


def run_simulation_with_snapshots(cfg: dict, n_snapshots: int = 50) -> tuple:
    """Run simulation and save snapshots at regular intervals for animation."""
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
    snapshot_interval = max(1, n_steps // n_snapshots)

    # Store snapshots
    times = [0.0]
    B_numerical = [np.array(state.B)]
    B_analytic = [np.array(config.analytic_solution(geometry, 0.0))]

    for step in range(n_steps):
        state = solver.step(state, dt, model, geometry)
        if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
            times.append(float(state.time))
            B_numerical.append(np.array(state.B))
            B_analytic.append(np.array(config.analytic_solution(geometry, state.time)))

    return times, B_numerical, B_analytic, geometry, config


ACCEPTANCE = {
    "l2_error": 0.01,              # 1% L2 error threshold
    "peak_amplitude_ratio": 0.98,  # 98% amplitude preservation
}


def create_2d_animation(times, B_numerical, B_analytic, geometry, save_path):
    """Create 2D side-by-side animation of analytic vs numerical solutions."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z_idx = geometry.nz // 2
    x = np.array(geometry.x_grid[:, :, z_idx])
    y = np.array(geometry.y_grid[:, :, z_idx])

    # Compute |B| for all snapshots
    B_mag_num = [np.sqrt(np.sum(B[:, :, z_idx, :]**2, axis=-1)) for B in B_numerical]
    B_mag_ana = [np.sqrt(np.sum(B[:, :, z_idx, :]**2, axis=-1)) for B in B_analytic]

    # Find global min/max for consistent colorbar
    vmin = 0
    vmax = max(np.max(B_mag_num[0]), np.max(B_mag_ana[0])) * 1.1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Initial plots
    levels = np.linspace(vmin, vmax, 21)
    im0 = axes[0].contourf(x, y, B_mag_ana[0], levels=levels, cmap="viridis")
    im1 = axes[1].contourf(x, y, B_mag_num[0], levels=levels, cmap="viridis")

    error = B_mag_num[0] - B_mag_ana[0]
    err_max = max(abs(np.min(error)), abs(np.max(error)), 0.01)
    err_levels = np.linspace(-err_max, err_max, 21)
    im2 = axes[2].contourf(x, y, error, levels=err_levels, cmap="RdBu_r")

    axes[0].set_title(f"Analytic |B| (t={times[0]:.3f}s)")
    axes[1].set_title(f"Numerical |B| (t={times[0]:.3f}s)")
    axes[2].set_title("Error (Num - Ana)")

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    plt.colorbar(im0, ax=axes[0], label="|B| [T]")
    plt.colorbar(im1, ax=axes[1], label="|B| [T]")
    plt.colorbar(im2, ax=axes[2], label="Error [T]")

    plt.tight_layout()

    def update(frame):
        for ax in axes:
            ax.clear()

        axes[0].contourf(x, y, B_mag_ana[frame], levels=levels, cmap="viridis")
        axes[1].contourf(x, y, B_mag_num[frame], levels=levels, cmap="viridis")

        error = B_mag_num[frame] - B_mag_ana[frame]
        axes[2].contourf(x, y, error, levels=err_levels, cmap="RdBu_r")

        axes[0].set_title(f"Analytic |B| (t={times[frame]:.3f}s)")
        axes[1].set_title(f"Numerical |B| (t={times[frame]:.3f}s)")
        axes[2].set_title("Error (Num - Ana)")

        for ax in axes:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")

        return axes

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(save_path, writer="pillow", fps=10)
    plt.close(fig)
    print(f"  Saved 2D animation: {save_path}")


def create_1d_animation(times, B_numerical, B_analytic, geometry, config, save_path):
    """Create 1D overlay animation showing |B| profiles along y=0 slice."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z_idx = geometry.nz // 2
    j_mid = geometry.ny // 2  # y=0 slice

    x = np.array(geometry.x_grid[:, j_mid, z_idx])

    # Compute |B| along y=0 for all snapshots
    B_mag_num = [np.sqrt(np.sum(B[:, j_mid, z_idx, :]**2, axis=-1)) for B in B_numerical]
    B_mag_ana = [np.sqrt(np.sum(B[:, j_mid, z_idx, :]**2, axis=-1)) for B in B_analytic]

    # Find global max for consistent y-axis
    ymax = max(np.max(B_mag_num[0]), np.max(B_mag_ana[0])) * 1.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: |B| profiles
    line_ana, = axes[0].plot(x, B_mag_ana[0], "b-", linewidth=2, label="Analytic")
    line_num, = axes[0].plot(x, B_mag_num[0], "r--", linewidth=2, label="Numerical")
    axes[0].set_xlim(x.min(), x.max())
    axes[0].set_ylim(0, ymax)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("|B| [T]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    title0 = axes[0].set_title(f"|B| along y=0 (t={times[0]:.3f}s)")

    # Right: Error profile
    error = np.array(B_mag_num[0]) - np.array(B_mag_ana[0])
    line_err, = axes[1].plot(x, error, "k-", linewidth=2)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlim(x.min(), x.max())
    err_max = max(abs(np.min(error)), abs(np.max(error)), 0.01) * 1.5
    axes[1].set_ylim(-err_max, err_max)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("Error [T]")
    axes[1].grid(True, alpha=0.3)
    title1 = axes[1].set_title("Error (Numerical - Analytic)")

    plt.tight_layout()

    def update(frame):
        line_ana.set_ydata(B_mag_ana[frame])
        line_num.set_ydata(B_mag_num[frame])
        title0.set_text(f"|B| along y=0 (t={times[frame]:.3f}s)")

        error = np.array(B_mag_num[frame]) - np.array(B_mag_ana[frame])
        line_err.set_ydata(error)

        return line_ana, line_num, line_err, title0, title1

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(save_path, writer="pillow", fps=10)
    plt.close(fig)
    print(f"  Saved 1D animation: {save_path}")


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

    print("Running simulation with snapshots for animation...")
    t_start = time.time()
    times, B_numerical, B_analytic, geometry, config = run_simulation_with_snapshots(cfg, n_snapshots=50)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Get final state for metrics
    B_final = jnp.array(B_numerical[-1])
    B_analytic_final = jnp.array(B_analytic[-1])
    B_initial = jnp.array(B_numerical[0])

    # Compute L2 error vs analytic
    l2_err = l2_error(B_final, B_analytic_final)

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

    B_mag_analytic = jnp.sqrt(jnp.sum(B_analytic_final[:, :, z_idx, :]**2, axis=-1))
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

    # Generate animations
    print("Generating animations...")
    report_dir = report.save()

    anim_2d_path = Path(report_dir) / "frozen_flux_2d_evolution.gif"
    anim_1d_path = Path(report_dir) / "frozen_flux_1d_evolution.gif"

    create_2d_animation(times, B_numerical, B_analytic, geometry, str(anim_2d_path))
    create_1d_animation(times, B_numerical, B_analytic, geometry, config, str(anim_1d_path))

    print()
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
