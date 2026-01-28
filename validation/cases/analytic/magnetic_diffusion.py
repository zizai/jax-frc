"""3D Magnetic Field Diffusion Validation (Cartesian Geometry, 2D-like)

Physics:
    This test validates the MHD solver against the exact solution
    for diffusion of a Gaussian magnetic field profile in the low magnetic
    Reynolds number limit (Rm << 1).

    Magnetic induction equation (no flow):
        ∂B/∂t = η∇²B

    where η is the magnetic diffusivity [m²/s], related to resistivity by
    η = 1/(μ₀σ) where σ is the electrical conductivity.

    Coordinate System:
        The MHD code uses 3D Cartesian (x, y, z) coordinates. For 2D-like
        behavior, we use a single-cell z-dimension (nz=1) with periodic boundary
        conditions. This effectively reduces the problem to 2D diffusion
        in the x-y plane.

    Initial condition (2D Gaussian in x-y plane):
        B_z(x, y, 0) = B_peak * exp(-(x² + y²) / (2σ²))

    Analytic solution (2D spreading Gaussian):
        σ_t² = σ² + 2ηt
        B_z(x, y, t) = B_peak * (σ²/σ_t²) * exp(-(x² + y²) / (2σ_t²))

    The Gaussian is in the x-y plane (uniform in z) to ensure div(B) = 0,
    since ∂Bz/∂z = 0. This is required for the MHD induction equation
    to correctly reduce to scalar diffusion.

    The Gaussian spreads and decreases in amplitude while conserving total
    magnetic flux. This is the magnetic analog of heat diffusion and serves
    as a fundamental test for resistive transport.

    The magnetic Reynolds number Rm = vL/η characterizes the ratio of
    convective to diffusive transport. This test uses v=0 (pure diffusion),
    so Rm = 0 << 1.

Models Tested:
    - ResistiveMHD: Basic resistive MHD with E = eta*J
    - ExtendedMHD: Extended MHD with Hall and electron pressure disabled

Reference:
    Standard diffusion equation theory; analogous to Carslaw & Jaeger,
    "Conduction of Heat in Solids", 2nd Ed., Ch. 2.
"""

import time
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.magnetic_diffusion import MagneticDiffusionConfiguration
from jax_frc.solvers.explicit import RK4Solver
from jax_frc.validation.metrics import l2_error, linf_error
from validation.utils.reporting import ValidationReport


# Case metadata
NAME = "magnetic_diffusion"
DESCRIPTION = "3D magnetic field diffusion test comparing ResistiveMHD and ExtendedMHD (Rm << 1)"

# Models to test
MODELS_TO_TEST = ["resistive_mhd", "extended_mhd"]

# Model display names and colors for plotting
MODEL_STYLES = {
    "resistive_mhd": {"label": "ResistiveMHD", "color": "blue", "linestyle": "-"},
    "extended_mhd": {"label": "ExtendedMHD", "color": "orange", "linestyle": "-"},
}


def setup_configuration() -> dict:
    """Return configuration parameters for this validation case."""
    eta = 1.26e-10  # Resistivity [Ω·m]
    MU0 = 1.2566e-6
    diffusivity = eta / MU0  # ~1e-4 m²/s

    nx = 64
    ny = 64
    nz = 1
    extent = 1.0

    dx = 2 * extent / nx
    dy = 2 * extent / ny
    dx_min = min(dx, dy)
    dt_max = 0.25 * dx_min**2 / diffusivity
    dt = dt_max * 0.5

    sigma = 0.1
    tau_diff = sigma**2 / (2 * diffusivity)
    t_end = 0.1 * tau_diff

    return {
        "B_peak": 1.0,
        "sigma": sigma,
        "eta": eta,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "extent": extent,
        "bc_x": "neumann",
        "bc_y": "neumann",
        "bc_z": "neumann",
        "t_end": t_end,
        "dt": dt,
    }


def run_simulation(cfg: dict, model_type: str) -> tuple:
    """Run the magnetic diffusion simulation with specified model.

    Args:
        cfg: Configuration dictionary from setup_configuration().
        model_type: Model type ("resistive_mhd" or "extended_mhd").

    Returns:
        Tuple of (final_state, geometry, config) after time integration.
    """
    config = MagneticDiffusionConfiguration(
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        extent=cfg["extent"],
        bc_x=cfg["bc_x"],
        bc_y=cfg["bc_y"],
        bc_z=cfg["bc_z"],
        B_peak=cfg["B_peak"],
        sigma=cfg["sigma"],
        eta=cfg["eta"],
        model_type=model_type,
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


# Acceptance criteria
ACCEPTANCE = {
    "l2_error": 0.01,
    "linf_error": 0.02,
}


def main() -> bool:
    """Run validation for all models and generate report."""
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    print()

    cfg = setup_configuration()
    print("Configuration:")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print()

    # Run simulations for each model
    results = {}
    geometry = None
    config = None

    for model_type in MODELS_TO_TEST:
        print(f"Testing model: {model_type}")
        t_start = time.time()
        final_state, geometry, config = run_simulation(cfg, model_type)
        t_sim = time.time() - t_start
        print(f"  Completed in {t_sim:.2f}s")

        # Extract B_z field (2D slice)
        z_idx = geometry.nz // 2
        Bz_sim_2d = final_state.B[:, :, z_idx, 2]

        results[model_type] = {
            "state": final_state,
            "Bz_2d": Bz_sim_2d,
            "time": t_sim,
        }
    print()

    # Compute analytic solution
    x_2d = geometry.x_grid[:, :, geometry.nz // 2]
    y_2d = geometry.y_grid[:, :, geometry.nz // 2]
    Bz_analytic_2d = config.analytic_solution_2d(x_2d, y_2d, cfg["t_end"])

    # Compute metrics for each model
    print("Results:")
    print(f"  {'Model':<15} {'L2 Error':<12} {'Linf Error':<12} {'Status'}")
    print("  " + "-" * 51)

    metrics = {}
    all_pass = True

    for model_type in MODELS_TO_TEST:
        Bz_sim = results[model_type]["Bz_2d"]
        l2_err = float(l2_error(Bz_sim, Bz_analytic_2d))
        linf_err = float(linf_error(Bz_sim, Bz_analytic_2d))

        l2_pass = l2_err < ACCEPTANCE["l2_error"]
        linf_pass = linf_err < ACCEPTANCE["linf_error"]
        model_pass = l2_pass and linf_pass
        all_pass = all_pass and model_pass

        status = "PASS" if model_pass else "FAIL"
        print(f"  {model_type:<15} {l2_err:<12.4g} {linf_err:<12.4g} {status}")

        results[model_type]["l2_error"] = l2_err
        results[model_type]["linf_error"] = linf_err
        results[model_type]["passed"] = model_pass

        metrics[f"{model_type}_l2"] = {
            "value": l2_err,
            "threshold": ACCEPTANCE["l2_error"],
            "passed": l2_pass,
            "description": f"{MODEL_STYLES[model_type]['label']} L2 error",
        }
        metrics[f"{model_type}_linf"] = {
            "value": linf_err,
            "threshold": ACCEPTANCE["linf_error"],
            "passed": linf_pass,
            "description": f"{MODEL_STYLES[model_type]['label']} Linf error",
        }
    print()

    # Create report
    total_time = sum(r["time"] for r in results.values())
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=metrics,
        overall_pass=all_pass,
        timing={"total_simulation": total_time},
    )

    # Create combined comparison plots
    y_center_idx = geometry.ny // 2
    x_center_idx = geometry.nx // 2
    x_1d = x_2d[:, y_center_idx]
    y_1d = y_2d[x_center_idx, :]
    Bz_analytic_x = Bz_analytic_2d[:, y_center_idx]
    Bz_analytic_y = Bz_analytic_2d[x_center_idx, :]

    # Plot 1: B_z profile at y=0 (x-slice)
    fig_x, ax_x = plt.subplots(figsize=(10, 6))
    ax_x.plot(x_1d, Bz_analytic_x, 'k--', linewidth=2, label='Analytic')
    for model_type in MODELS_TO_TEST:
        style = MODEL_STYLES[model_type]
        Bz_sim_x = results[model_type]["Bz_2d"][:, y_center_idx]
        ax_x.plot(x_1d, Bz_sim_x, color=style["color"], linestyle=style["linestyle"],
                  linewidth=1.5, label=style["label"])
    ax_x.set_xlabel("x [m]")
    ax_x.set_ylabel("B_z [T]")
    ax_x.set_title(f"B_z Profile at y=0, t={cfg['t_end']:.1e}s")
    ax_x.legend()
    ax_x.grid(True, alpha=0.3)
    fig_x.tight_layout()
    report.add_plot(fig_x, name="Bz_comparison_x")

    # Plot 2: B_z profile at x=0 (y-slice)
    fig_y, ax_y = plt.subplots(figsize=(10, 6))
    ax_y.plot(y_1d, Bz_analytic_y, 'k--', linewidth=2, label='Analytic')
    for model_type in MODELS_TO_TEST:
        style = MODEL_STYLES[model_type]
        Bz_sim_y = results[model_type]["Bz_2d"][x_center_idx, :]
        ax_y.plot(y_1d, Bz_sim_y, color=style["color"], linestyle=style["linestyle"],
                  linewidth=1.5, label=style["label"])
    ax_y.set_xlabel("y [m]")
    ax_y.set_ylabel("B_z [T]")
    ax_y.set_title(f"B_z Profile at x=0, t={cfg['t_end']:.1e}s")
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)
    fig_y.tight_layout()
    report.add_plot(fig_y, name="Bz_comparison_y")

    # Plot 3: Error plot (simulation - analytic)
    fig_err, ax_err = plt.subplots(figsize=(10, 6))
    for model_type in MODELS_TO_TEST:
        style = MODEL_STYLES[model_type]
        Bz_sim_y = results[model_type]["Bz_2d"][x_center_idx, :]
        error_y = Bz_sim_y - Bz_analytic_y
        ax_err.plot(y_1d, error_y, color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.5, label=style["label"])
    ax_err.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax_err.set_xlabel("y [m]")
    ax_err.set_ylabel("Error [T]")
    ax_err.set_title("B_z Error at x=0 (Simulation - Analytic)")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)
    fig_err.tight_layout()
    report.add_plot(fig_err, name="Bz_error")

    # Save report
    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    # Print result
    if all_pass:
        print("PASS: All models meet acceptance criteria")
    else:
        print("FAIL: Some models did not meet acceptance criteria")
        for model_type in MODELS_TO_TEST:
            if not results[model_type]["passed"]:
                l2 = results[model_type]["l2_error"]
                linf = results[model_type]["linf_error"]
                print(f"  - {model_type}: L2={l2:.4g}, Linf={linf:.4g}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
