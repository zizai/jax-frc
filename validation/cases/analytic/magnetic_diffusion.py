"""2D Magnetic Field Diffusion Validation (Slab Geometry)

Physics:
    This test validates the MHD solver against the exact solution
    for 2D diffusion of a Gaussian magnetic field profile in the low magnetic
    Reynolds number limit (Rm << 1).

    Magnetic induction equation (no flow):
        ∂B/∂t = η∇²B

    where η is the magnetic diffusivity [m²/s], related to resistivity by
    η = 1/(μ₀σ) where σ is the electrical conductivity.

    Coordinate Transformation:
        The MHD code uses cylindrical 2D (r,z) coordinates, while the analytic
        solution is for 2D Cartesian diffusion. By centering the domain at
        large r (r_center >> sigma), the cylindrical Laplacian approximates
        Cartesian: ∇²B_z ≈ ∂²B_z/∂r² + ∂²B_z/∂z² (the (1/r)∂B_z/∂r term
        becomes negligible when r >> sigma).

    Initial condition (2D Gaussian):
        B_z(r, z, 0) = B_peak * exp(-((r-r₀)² + (z-z₀)²) / (2σ²))

    Analytic solution (2D spreading Gaussian):
        σ_t² = σ² + 2ηt
        B_z(r, z, t) = B_peak * (σ²/σ_t²) * exp(-((r-r₀)² + (z-z₀)²) / (2σ_t²))

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
from jax_frc.solvers.explicit import EulerSolver
from jax_frc.validation.metrics import l2_error, linf_error
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error


# Case metadata
NAME = "magnetic_diffusion"
DESCRIPTION = "2D magnetic field diffusion test in slab geometry with analytic solution (Rm << 1)"


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

    # Grid parameters
    nr = 64
    nz = 64
    r_min = 1.0
    r_max = 2.0
    dr = (r_max - r_min) / nr
    dz = 2.0 / nz  # z_extent = 1.0, so z goes from -1 to 1

    # CFL constraint for diffusion: dt < dx^2 / (2*D)
    dx_min = min(dr, dz)
    dt_max = 0.25 * dx_min**2 / diffusivity
    dt = dt_max * 0.5  # Safety factor

    # Run time - simulate for 0.1 diffusion times
    sigma = 0.1
    tau_diff = sigma**2 / (2 * diffusivity)
    t_end = 0.1 * tau_diff

    return {
        "B_peak": 1.0,       # Peak magnetic field [T]
        "sigma": sigma,      # Initial Gaussian width [m] (should be << r_min)
        "eta": eta,          # Magnetic resistivity [Ω·m]
        "nr": nr,            # Radial resolution
        "nz": nz,            # Axial resolution
        "r_min": r_min,      # Inner radius [m] - far from axis
        "r_max": r_max,      # Outer radius [m]
        "z_extent": 1.0,     # Domain: z ∈ [-z_extent, z_extent] [m]
        "r_center": 1.5,     # Radial center of Gaussian [m]
        "z_center": 0.0,     # Axial center of Gaussian [m]
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
        B_peak=cfg["B_peak"],
        sigma=cfg["sigma"],
        eta=cfg["eta"],
        nr=cfg["nr"],
        nz=cfg["nz"],
        r_min=cfg["r_min"],
        r_max=cfg["r_max"],
        z_extent=cfg["z_extent"],
        r_center=cfg["r_center"],
        z_center=cfg["z_center"],
    )

    # Create geometry and initial state
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    # Create Euler solver for explicit time stepping
    solver = EulerSolver()

    # Time stepping
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, geometry, config


# Acceptance criteria (adjusted for coordinate transformation approximation)
ACCEPTANCE = {
    "l2_error": 0.10,      # 10% relative L2 error threshold (higher due to cylindrical approx)
    "linf_error": 0.15,    # Max pointwise error threshold [T]
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

    # Extract full 2D B_z field
    r = geometry.r_grid
    z = geometry.z_grid
    Bz_sim = final_state.B[:, :, 2]

    # Compute analytic solution at final time with coordinate transformation
    Bz_analytic = config.analytic_solution(r, z, cfg["t_end"])

    # Compute metrics on full 2D field
    l2_err = l2_error(Bz_sim, Bz_analytic)
    linf_err = linf_error(Bz_sim, Bz_analytic)

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
            "description": "Relative L2 error vs analytic solution (2D)",
        },
        "linf_error": {
            "value": float(linf_err),
            "threshold": ACCEPTANCE["linf_error"],
            "passed": linf_pass,
            "description": "Maximum pointwise B_z error [T] (2D)",
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
    # Radial slice at z_center
    z_center_idx = geometry.nz // 2
    r_1d = r[:, z_center_idx]
    Bz_sim_r = Bz_sim[:, z_center_idx]
    Bz_analytic_r = Bz_analytic[:, z_center_idx]

    fig_comparison_r = plot_comparison(
        x=r_1d,
        actual=Bz_sim_r,
        expected=Bz_analytic_r,
        labels=("Simulation", "Analytic"),
        title=f"B_z Profile at z=0, t={cfg['t_end']:.1e}s",
        xlabel="r [m]",
        ylabel="B_z [T]",
    )
    report.add_plot(fig_comparison_r, name="Bz_comparison_r")

    # Axial slice at r_center
    r_center_idx = geometry.nr // 2
    z_1d = z[r_center_idx, :]
    Bz_sim_z = Bz_sim[r_center_idx, :]
    Bz_analytic_z = Bz_analytic[r_center_idx, :]

    fig_comparison_z = plot_comparison(
        x=z_1d,
        actual=Bz_sim_z,
        expected=Bz_analytic_z,
        labels=("Simulation", "Analytic"),
        title=f"B_z Profile at r=r_center, t={cfg['t_end']:.1e}s",
        xlabel="z [m]",
        ylabel="B_z [T]",
    )
    report.add_plot(fig_comparison_z, name="Bz_comparison_z")

    fig_error = plot_error(
        x=z_1d,
        actual=Bz_sim_z,
        expected=Bz_analytic_z,
        title="B_z Error at r=r_center (Simulation - Analytic)",
        xlabel="z [m]",
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
