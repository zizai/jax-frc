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

Reference:
    Standard diffusion equation theory; analogous to Carslaw & Jaeger,
    "Conduction of Heat in Solids", 2nd Ed., Ch. 2.
"""

import time
import sys
from pathlib import Path

import jax.numpy as jnp

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.magnetic_diffusion import MagneticDiffusionConfiguration
from jax_frc.solvers.explicit import RK4Solver
from jax_frc.validation.metrics import l2_error, linf_error
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error


# Case metadata
NAME = "magnetic_diffusion"
DESCRIPTION = "3D magnetic field diffusion test in Cartesian geometry with analytic solution (Rm << 1)"


def setup_configuration() -> dict:
    """Return configuration parameters for this validation case.

    Returns:
        Dictionary with all physics and numerical parameters.
    """
    # Note: eta is resistivity [Ω·m], diffusivity D = eta/mu_0
    # For diffusivity D = 1e-4 m²/s, eta = D * mu_0 ≈ 1.26e-10 Ω·m
    eta = 1.26e-10  # Resistivity [Ω·m]
    MU0 = 1.2566e-6
    diffusivity = eta / MU0  # ~1e-4 m²/s

    # Grid parameters (3D Cartesian with thin z for 2D-like behavior)
    nx = 64
    ny = 64
    nz = 1   # Single cell in z (pseudo-dimension for 2D simulation)
    extent = 1.0  # Domain: [-extent, extent] in each direction

    dx = 2 * extent / nx
    dy = 2 * extent / ny

    # CFL constraint for diffusion: dt < dx^2 / (2*D)
    dx_min = min(dx, dy)
    dt_max = 0.25 * dx_min**2 / diffusivity
    dt = dt_max * 0.5  # Safety factor

    # Run time - simulate for 0.1 diffusion times
    sigma = 0.1
    tau_diff = sigma**2 / (2 * diffusivity)
    t_end = 0.1 * tau_diff

    return {
        "B_peak": 1.0,       # Peak magnetic field [T]
        "sigma": sigma,      # Initial Gaussian width [m]
        "eta": eta,          # Magnetic resistivity [Ω·m]
        "nx": nx,            # X resolution
        "ny": ny,            # Y resolution
        "nz": nz,            # Z resolution (thin for 2D-like)
        "extent": extent,    # Domain extent [m]
        "t_end": t_end,      # End time [s]
        "dt": dt,            # Time step [s] (CFL-constrained)
    }


def run_simulation(cfg: dict) -> tuple:
    """Run the magnetic diffusion simulation.

    Args:
        cfg: Configuration dictionary from setup_configuration().

    Returns:
        Tuple of (final_state, geometry, config) after time integration.
    """
    # Build configuration with specified parameters
    config = MagneticDiffusionConfiguration(
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        extent=cfg["extent"],
        B_peak=cfg["B_peak"],
        sigma=cfg["sigma"],
        eta=cfg["eta"],
    )

    # Create geometry and initial state
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    # Create Euler solver for explicit time stepping
    solver = RK4Solver()

    # Time stepping
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, geometry, config


# Acceptance criteria
ACCEPTANCE = {
    "l2_error": 0.01,      # 1% relative L2 error threshold
    "linf_error": 0.02,    # 2% max pointwise error threshold
}


def main() -> bool:
    """Run validation and generate report.

    Returns:
        True if all acceptance criteria pass, False otherwise.
    """
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    print()

    # Setup
    cfg = setup_configuration()
    print("Configuration:")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print()

    # Run simulation with timing
    print("Running simulation...")
    t_start = time.time()
    final_state, geometry, config = run_simulation(cfg)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Extract 3D grids and B_z field
    x = geometry.x_grid  # Shape: (nx, ny, nz)
    y = geometry.y_grid
    z = geometry.z_grid
    Bz_sim = final_state.B[:, :, :, 2]  # 3D array

    # For 2D-like comparison, take slice at z=0 (middle of thin z-dimension)
    z_idx = geometry.nz // 2

    # Extract 2D slices for comparison
    x_2d = x[:, :, z_idx]  # Shape: (nx, ny)
    y_2d = y[:, :, z_idx]
    Bz_sim_2d = Bz_sim[:, :, z_idx]

    # Compute 2D analytic solution at final time
    Bz_analytic_2d = config.analytic_solution_2d(x_2d, y_2d, cfg["t_end"])

    # Compute metrics on 2D slice
    l2_err = l2_error(Bz_sim_2d, Bz_analytic_2d)
    linf_err = linf_error(Bz_sim_2d, Bz_analytic_2d)

    print("Metrics:")
    print(f"  L2 error:   {l2_err:.4g} (threshold: {ACCEPTANCE['l2_error']})")
    print(f"  Linf error: {linf_err:.4g} (threshold: {ACCEPTANCE['linf_error']})")
    print()

    # Check acceptance
    l2_pass = l2_err < ACCEPTANCE["l2_error"]
    linf_pass = linf_err < ACCEPTANCE["linf_error"]
    overall_pass = l2_pass and linf_pass

    # Build metrics dictionary for report
    metrics = {
        "l2_error": {
            "value": float(l2_err),
            "threshold": ACCEPTANCE["l2_error"],
            "passed": l2_pass,
            "description": "Relative L2 error vs 2D analytic solution",
        },
        "linf_error": {
            "value": float(linf_err),
            "threshold": ACCEPTANCE["linf_error"],
            "passed": linf_pass,
            "description": "Maximum pointwise B_z error [T]",
        },
    }

    # Create report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=metrics,
        overall_pass=overall_pass,
        timing={"simulation": t_sim},
    )

    # Add plots - use center slices for visualization
    # X-slice at y=0 (center of domain)
    y_center_idx = geometry.ny // 2
    x_1d = x_2d[:, y_center_idx]
    Bz_sim_x = Bz_sim_2d[:, y_center_idx]
    Bz_analytic_x = Bz_analytic_2d[:, y_center_idx]

    fig_comparison_x = plot_comparison(
        x=x_1d,
        actual=Bz_sim_x,
        expected=Bz_analytic_x,
        labels=("Simulation", "Analytic"),
        title=f"B_z Profile at y=0, t={cfg['t_end']:.1e}s",
        xlabel="x [m]",
        ylabel="B_z [T]",
    )
    report.add_plot(fig_comparison_x, name="Bz_comparison_x")

    # Y-slice at x=0 (center of domain)
    x_center_idx = geometry.nx // 2
    y_1d = y_2d[x_center_idx, :]
    Bz_sim_y = Bz_sim_2d[x_center_idx, :]
    Bz_analytic_y = Bz_analytic_2d[x_center_idx, :]

    fig_comparison_y = plot_comparison(
        x=y_1d,
        actual=Bz_sim_y,
        expected=Bz_analytic_y,
        labels=("Simulation", "Analytic"),
        title=f"B_z Profile at x=0, t={cfg['t_end']:.1e}s",
        xlabel="y [m]",
        ylabel="B_z [T]",
    )
    report.add_plot(fig_comparison_y, name="Bz_comparison_y")

    fig_error = plot_error(
        x=y_1d,
        actual=Bz_sim_y,
        expected=Bz_analytic_y,
        title="B_z Error at x=0 (Simulation - Analytic)",
        xlabel="y [m]",
        ylabel="Error [T]",
    )
    report.add_plot(fig_error, name="Bz_error")

    # Save report
    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    # Print result
    if overall_pass:
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: Some acceptance criteria not met")
        if not l2_pass:
            print(f"  - L2 error {l2_err:.4g} exceeds threshold {ACCEPTANCE['l2_error']}")
        if not linf_pass:
            print(f"  - Linf error {linf_err:.4g} exceeds threshold {ACCEPTANCE['linf_error']}")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
