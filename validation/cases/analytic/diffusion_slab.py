"""1D Slab Heat Diffusion Validation

Physics:
    This test validates the heat conduction solver against the exact solution
    for 1D diffusion of a Gaussian temperature profile.

    Heat equation:
        dT/dt = kappa * d^2T/dz^2

    where kappa is the thermal diffusivity [m^2/s].

    Initial condition:
        T(z, 0) = T_peak * exp(-z^2 / (2*sigma^2)) + T_base

    Analytic solution:
        sigma_t^2 = sigma^2 + 2*kappa*t
        T(z, t) = T_peak * sqrt(sigma^2 / sigma_t^2) * exp(-z^2 / (2*sigma_t^2)) + T_base

    The Gaussian spreads and decreases in amplitude while conserving total heat.
    This is a fundamental test for any thermal transport implementation.

Reference:
    Carslaw & Jaeger, "Conduction of Heat in Solids", 2nd Ed., Ch. 2.
"""

import time
import sys
from pathlib import Path

import jax.numpy as jnp

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.analytic import SlabDiffusionConfiguration
from jax_frc.solvers.semi_implicit import SemiImplicitSolver
from jax_frc.validation.metrics import l2_error, linf_error
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error


# Case metadata
NAME = "diffusion_slab"
DESCRIPTION = "1D heat diffusion test with analytic solution"


def setup_configuration() -> dict:
    """Return configuration parameters for this validation case.

    Returns:
        Dictionary with all physics and numerical parameters.
    """
    return {
        "T_peak": 200.0,    # Peak temperature [eV]
        "T_base": 50.0,     # Background temperature [eV]
        "sigma": 0.3,       # Initial Gaussian width [m]
        "kappa": 1.0e-3,    # Thermal diffusivity [m^2/s]
        "nz": 64,           # Grid points in z
        "t_end": 1.0e-4,    # End time [s]
        "dt": 1.0e-6,       # Time step [s]
    }


def analytic_solution(z: jnp.ndarray, t: float, cfg: dict) -> jnp.ndarray:
    """Compute exact Gaussian diffusion solution.

    Args:
        z: Array of z positions.
        t: Time at which to evaluate the solution.
        cfg: Configuration dictionary with T_peak, T_base, sigma, kappa.

    Returns:
        Temperature array T(z, t) from the exact analytic solution.
    """
    sigma = cfg["sigma"]
    kappa = cfg["kappa"]
    T_peak = cfg["T_peak"]
    T_base = cfg["T_base"]

    # Effective width grows with time due to diffusion
    sigma_t_sq = sigma**2 + 2 * kappa * t

    # Amplitude decreases to conserve total heat
    amplitude = T_peak * jnp.sqrt(sigma**2 / sigma_t_sq)

    return amplitude * jnp.exp(-z**2 / (2 * sigma_t_sq)) + T_base


def run_simulation(cfg: dict) -> tuple:
    """Run the heat diffusion simulation.

    Args:
        cfg: Configuration dictionary from setup_configuration().

    Returns:
        Tuple of (final_state, geometry) after time integration.
    """
    # Build configuration with specified parameters
    config = SlabDiffusionConfiguration(
        T_peak=cfg["T_peak"],
        T_base=cfg["T_base"],
        sigma=cfg["sigma"],
        kappa=cfg["kappa"],
        nz=cfg["nz"],
    )

    # Create geometry and initial state
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    # Create semi-implicit solver with high damping to suppress Hall instability
    # This allows isolated testing of thermal conduction without B field feedback
    solver = SemiImplicitSolver(damping_factor=1e20)

    # Time stepping
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, geometry


# Acceptance criteria
ACCEPTANCE = {
    "l2_error": 0.1,      # Relative L2 error threshold
    "linf_error": 0.15,   # Max pointwise error threshold (in eV)
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
    final_state, geometry = run_simulation(cfg)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Extract z-axis temperature profile (take middle r-slice)
    z = geometry.z_grid[0, :]  # 1D z coordinates
    T_sim = final_state.T[geometry.nr // 2, :]  # Temperature at mid-radius

    # Compute analytic solution at final time
    T_analytic = analytic_solution(z, cfg["t_end"], cfg)

    # Compute metrics
    l2_err = l2_error(T_sim, T_analytic)
    linf_err = linf_error(T_sim, T_analytic)

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
            "value": l2_err,
            "threshold": ACCEPTANCE["l2_error"],
            "passed": l2_pass,
            "description": "Relative L2 error vs analytic solution",
        },
        "linf_error": {
            "value": linf_err,
            "threshold": ACCEPTANCE["linf_error"],
            "passed": linf_pass,
            "description": "Maximum pointwise temperature error [eV]",
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

    # Add plots
    fig_comparison = plot_comparison(
        x=z,
        actual=T_sim,
        expected=T_analytic,
        labels=("Simulation", "Analytic"),
        title=f"Temperature Profile at t={cfg['t_end']:.1e}s",
        xlabel="z [m]",
        ylabel="Temperature [eV]",
    )
    report.add_plot(fig_comparison, name="temperature_comparison")

    fig_error = plot_error(
        x=z,
        actual=T_sim,
        expected=T_analytic,
        title="Temperature Error (Simulation - Analytic)",
        xlabel="z [m]",
        ylabel="Error [eV]",
    )
    report.add_plot(fig_error, name="temperature_error")

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
