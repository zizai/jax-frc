"""Belova Case 4: Large FRC with Compression (Qualitative)

Physics:
    This test validates qualitative behavior for compression-driven FRC
    (Field-Reversed Configuration) merging simulations based on Belova et al.
    The simulation models counter-helicity merger of two large FRC plasmoids
    where merging is driven by external magnetic compression rather than
    initial axial velocity.

    Key physics:
        - Two FRCs with opposite toroidal field (counter-helicity)
        - Large FRC parameters same as Case 1, but with compression
        - Mirror ratio: 1.5, ramp time: 19 tA
        - Resistive MHD with Chodura anomalous resistivity
        - Conducting wall boundary conditions
        - Time-dependent mirror field drives merging

    FRC parameters (Belova et al. Fig. 6-7):
        S* = 25.6 (kinetic parameter)
        E = 2.9 (elongation)
        xs = 0.69 (normalized separatrix radius)
        beta_s = 0.2 (separatrix beta)

    Expected outcome:
        Complete merge by ~20-25 tA. Unlike Case 1 where the large FRCs form
        a doublet, the external compression forces the FRCs together and
        enables complete merging.

    This is a QUALITATIVE validation - there is no analytic solution. We verify:
        - Numerical stability (no NaN/Inf)
        - Flux conservation within tolerances (15% allowed due to compression)
        - Energy remains bounded throughout evolution

Reference:
    Belova et al., "Numerical study of FRC merging for MRX", Phys. Plasmas
    (2003), Figures 6-7.
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

from jax_frc.configurations.frc_merging import BelovaCase4Configuration
from jax_frc.solvers import Solver
from validation.utils.reporting import ValidationReport


# Case metadata
NAME = "belova_case4"
DESCRIPTION = "Large FRC with compression (Belova et al. Fig. 6-7)"


def setup_configuration(quick_test: bool = False) -> dict:
    """Return configuration parameters for this validation case.

    Args:
        quick_test: If True, use reduced parameters for faster testing.

    Returns:
        Dictionary with all physics and numerical parameters.
    """
    if quick_test:
        return {
            "model_type": "resistive_mhd",
            "eta_0": 1e-6,
            "eta_anom": 1e-3,
            "t_end": 0.1,  # Very short for quick test
            "dt": 0.001,
            "nr": 32,
            "nz": 64,
        }

    return {
        "model_type": "resistive_mhd",
        "eta_0": 1e-6,
        "eta_anom": 1e-3,
        "t_end": 40.0,
        "dt": 0.001,
        "nr": 64,
        "nz": 512,
    }


def run_simulation(cfg: dict) -> tuple:
    """Run the FRC merging simulation.

    Args:
        cfg: Configuration dictionary from setup_configuration().

    Returns:
        Tuple of (final_state, geometry, history) where history is a dict
        containing time traces and snapshots.
    """
    # Build configuration with specified parameters
    config = BelovaCase4Configuration(
        model_type=cfg["model_type"],
        eta_0=cfg["eta_0"],
        eta_anom=cfg["eta_anom"],
        nr=cfg["nr"],
        nz=cfg["nz"],
    )

    # Create geometry and initial state
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    # Create solver - use semi-implicit for stability
    solver = Solver.create({"type": "semi_implicit"})

    # Time stepping parameters
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    # History capture - record at intervals
    output_interval = max(1, n_steps // 100)  # ~100 snapshots
    history = {
        "times": [],
        "total_flux": [],
        "total_energy": [],
        "psi_snapshots": [],
        "snapshot_times": [],
    }

    # Record initial state
    initial_flux = compute_total_flux(state, geometry)
    initial_energy = compute_total_energy(state, geometry)
    history["times"].append(0.0)
    history["total_flux"].append(float(initial_flux))
    history["total_energy"].append(float(initial_energy))
    history["psi_snapshots"].append(state.psi)
    history["snapshot_times"].append(0.0)

    # Time integration loop
    for step_idx in range(n_steps):
        state = solver.step(state, dt, model, geometry)

        # Check for early termination due to instability
        if not check_stability(state):
            print(f"  WARNING: Numerical instability detected at step {step_idx}")
            break

        # Record history at intervals
        if (step_idx + 1) % output_interval == 0:
            current_time = (step_idx + 1) * dt
            flux = compute_total_flux(state, geometry)
            energy = compute_total_energy(state, geometry)

            history["times"].append(current_time)
            history["total_flux"].append(float(flux))
            history["total_energy"].append(float(energy))

            # Store psi snapshots at coarser intervals (every 10th recording)
            if len(history["times"]) % 10 == 0:
                history["psi_snapshots"].append(state.psi)
                history["snapshot_times"].append(current_time)

        # Progress indicator
        progress_interval = max(1, n_steps // 10)
        if (step_idx + 1) % progress_interval == 0:
            print(f"    Step {step_idx + 1}/{n_steps} (t={state.time:.2f})")

    # Convert lists to arrays for easier processing
    history["times"] = jnp.array(history["times"])
    history["total_flux"] = jnp.array(history["total_flux"])
    history["total_energy"] = jnp.array(history["total_energy"])

    return state, geometry, history


def compute_total_flux(state, geometry) -> float:
    """Compute total poloidal flux integrated over domain.

    Args:
        state: Current simulation state.
        geometry: Computational geometry.

    Returns:
        Total flux (integral of psi over r*dr*dz).
    """
    dr = (geometry.r_max - geometry.r_min) / geometry.nr
    dz = (geometry.z_max - geometry.z_min) / geometry.nz
    r = geometry.r_grid

    # Integrate psi weighted by r (cylindrical coordinates)
    return jnp.sum(state.psi * r * dr * dz)


def compute_total_energy(state, geometry) -> float:
    """Compute total energy (magnetic + thermal).

    Args:
        state: Current simulation state.
        geometry: Computational geometry.

    Returns:
        Total energy integrated over domain.
    """
    dr = (geometry.r_max - geometry.r_min) / geometry.nr
    dz = (geometry.z_max - geometry.z_min) / geometry.nz
    r = geometry.r_grid

    # Magnetic energy: |B|^2 / 2
    B_mag_sq = jnp.sum(state.B**2, axis=-1)
    E_mag = 0.5 * B_mag_sq

    # Thermal energy: p / (gamma - 1), using gamma = 5/3
    gamma = 5.0 / 3.0
    E_thermal = state.p / (gamma - 1.0)

    # Total energy integrated over volume
    E_total = E_mag + E_thermal
    return jnp.sum(E_total * r * dr * dz)


# Acceptance criteria for qualitative validation
ACCEPTANCE = {
    "no_numerical_instability": {
        "description": "No NaN or Inf values in solution",
    },
    "flux_conservation": {
        "threshold": 0.15,  # 15% drift allowed (compression allows more variation)
        "description": "Total flux conserved within 15% (compression allows more variation)",
    },
    "energy_bounded": {
        "description": "Total energy remains bounded (no runaway growth)",
    },
}


def check_stability(state) -> bool:
    """Check if state contains any NaN or Inf values.

    Args:
        state: Simulation state to check.

    Returns:
        True if state is numerically stable (no NaN/Inf), False otherwise.
    """
    fields = [state.psi, state.n, state.p, state.T, state.B, state.E, state.v]
    for field in fields:
        if jnp.any(jnp.isnan(field)) or jnp.any(jnp.isinf(field)):
            return False
    return True


def check_flux_conservation(history, threshold: float) -> tuple[bool, float]:
    """Check if total flux is conserved within threshold.

    Args:
        history: History dictionary containing flux time trace.
        threshold: Maximum allowed relative change (e.g., 0.1 for 10%).

    Returns:
        Tuple of (passed, relative_drift) where passed is bool and
        relative_drift is the maximum relative change from initial.
    """
    flux = history["total_flux"]
    initial_flux = flux[0]

    if jnp.abs(initial_flux) < 1e-10:
        # Avoid division by near-zero
        return True, 0.0

    relative_change = jnp.abs((flux - initial_flux) / initial_flux)
    max_drift = float(jnp.max(relative_change))

    passed = max_drift < threshold
    return passed, max_drift


def check_energy_bounded(history) -> tuple[bool, float]:
    """Check if total energy remains bounded (no runaway growth).

    Energy should not grow by more than 10x during the simulation.

    Args:
        history: History dictionary containing energy time trace.

    Returns:
        Tuple of (passed, max_ratio) where passed is bool and
        max_ratio is the maximum energy / initial energy.
    """
    energy = history["total_energy"]
    initial_energy = energy[0]

    if jnp.abs(initial_energy) < 1e-10:
        # If starting energy is near zero, check absolute growth
        max_energy = float(jnp.max(jnp.abs(energy)))
        return max_energy < 1.0, max_energy

    max_ratio = float(jnp.max(energy / initial_energy))
    passed = max_ratio < 10.0  # Energy shouldn't grow more than 10x

    return passed, max_ratio


def plot_time_trace(times, values, title: str, ylabel: str, figsize=(10, 4)):
    """Create a time trace plot.

    Args:
        times: Array of time values.
        values: Array of corresponding values to plot.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, values, 'b-', linewidth=1.5)
    ax.set_xlabel('Time [Alfven times]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main(quick_test: bool = False) -> bool:
    """Run validation and generate report.

    Args:
        quick_test: If True, use reduced parameters for faster testing.

    Returns:
        True if all acceptance criteria pass, False otherwise.
    """
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    if quick_test:
        print("  (QUICK TEST MODE - reduced parameters)")
    print()

    # Setup
    cfg = setup_configuration(quick_test=quick_test)
    print("Configuration:")
    for key, val in cfg.items():
        print(f"  {key}: {val}")
    print()

    # Run simulation with timing
    print("Running simulation...")
    t_start = time.time()
    final_state, geometry, history = run_simulation(cfg)
    t_sim = time.time() - t_start
    print(f"  Completed in {t_sim:.2f}s")
    print()

    # Check acceptance criteria
    print("Checking acceptance criteria...")

    # 1. Numerical stability
    stability_pass = check_stability(final_state)
    print(f"  Numerical stability: {'PASS' if stability_pass else 'FAIL'}")

    # 2. Flux conservation
    flux_threshold = ACCEPTANCE["flux_conservation"]["threshold"]
    flux_pass, flux_drift = check_flux_conservation(history, flux_threshold)
    print(f"  Flux conservation: {'PASS' if flux_pass else 'FAIL'} "
          f"(drift: {flux_drift:.4g}, threshold: {flux_threshold})")

    # 3. Energy bounded
    energy_pass, energy_ratio = check_energy_bounded(history)
    print(f"  Energy bounded: {'PASS' if energy_pass else 'FAIL'} "
          f"(max ratio: {energy_ratio:.4g}, threshold: 10.0)")
    print()

    overall_pass = stability_pass and flux_pass and energy_pass

    # Build metrics dictionary for report
    metrics = {
        "numerical_stability": {
            "value": "stable" if stability_pass else "unstable",
            "passed": stability_pass,
            "description": ACCEPTANCE["no_numerical_instability"]["description"],
        },
        "flux_conservation": {
            "value": flux_drift,
            "threshold": flux_threshold,
            "passed": flux_pass,
            "description": ACCEPTANCE["flux_conservation"]["description"],
        },
        "energy_bounded": {
            "value": energy_ratio,
            "threshold": 10.0,
            "passed": energy_pass,
            "description": ACCEPTANCE["energy_bounded"]["description"],
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

    # Add flux evolution plot
    fig_flux = plot_time_trace(
        times=history["times"],
        values=history["total_flux"],
        title="Total Poloidal Flux Evolution",
        ylabel="Total Flux [Wb]",
    )
    report.add_plot(fig_flux, name="flux_evolution")
    plt.close(fig_flux)

    # Add energy evolution plot
    fig_energy = plot_time_trace(
        times=history["times"],
        values=history["total_energy"],
        title="Total Energy Evolution",
        ylabel="Total Energy [J]",
    )
    report.add_plot(fig_energy, name="energy_evolution")
    plt.close(fig_energy)

    # Add normalized flux plot (for conservation check)
    initial_flux = history["total_flux"][0]
    if jnp.abs(initial_flux) > 1e-10:
        normalized_flux = history["total_flux"] / initial_flux
        fig_flux_norm = plot_time_trace(
            times=history["times"],
            values=normalized_flux,
            title="Normalized Flux (relative to initial)",
            ylabel="Flux / Initial Flux",
        )
        report.add_plot(fig_flux_norm, name="flux_normalized")
        plt.close(fig_flux_norm)

    # Save report
    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    # Print result
    if overall_pass:
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: Some acceptance criteria not met")
        if not stability_pass:
            print("  - Numerical instability detected (NaN/Inf values)")
        if not flux_pass:
            print(f"  - Flux drift {flux_drift:.4g} exceeds threshold {flux_threshold}")
        if not energy_pass:
            print(f"  - Energy ratio {energy_ratio:.4g} exceeds threshold 10.0")

    return overall_pass


if __name__ == "__main__":
    # Check for --quick flag
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
