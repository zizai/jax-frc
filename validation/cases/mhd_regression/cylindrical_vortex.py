"""Orszag-Tang Vortex in Cylindrical Annulus

Physics:
    This test validates MHD turbulence dynamics using the Orszag-Tang vortex
    problem adapted to cylindrical coordinates in an annulus geometry. The
    simulation tests shock-capturing, current sheet formation, and energy
    conservation in compressible MHD.

    Key physics:
        - Initial velocity: vr = -v0*sin(z), vz = v0*sin(2*pi*r_norm)
        - Initial magnetic field: Br = -B0*sin(z), Bz = B0*sin(4*pi*r_norm)
        - Nonlinear evolution creates current sheets and shocks
        - Energy should be approximately conserved (resistive dissipation small)

    Orszag-Tang parameters:
        v0 = 1.0 (velocity amplitude)
        B0 = 1.0 (magnetic field amplitude)
        rho0 = 25/(36*pi) (initial density)
        p0 = 5/(12*pi) (initial pressure)
        gamma = 5/3 (adiabatic index)

    Expected outcomes:
        - Current sheets form at t ~ 0.25
        - Peak current density at t ~ 0.48
        - Total energy conserved within 1% (resistive MHD)

    This is a REGRESSION validation - we track that the code produces consistent
    results and satisfies conservation properties. We verify:
        - Numerical stability (no NaN/Inf)
        - Energy conservation within tolerance
        - Time evolution follows expected pattern

Reference:
    Orszag & Tang, "Small-scale structure of two-dimensional MHD turbulence",
    J. Fluid Mech. (1979)
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

from jax_frc.configurations.orszag_tang import OrszagTangConfiguration
from jax_frc.solvers import Solver
from validation.utils.reporting import ValidationReport


# Case metadata
NAME = "cylindrical_vortex"
DESCRIPTION = "Orszag-Tang vortex in cylindrical annulus"


def setup_configuration(quick_test: bool = False) -> dict:
    """Return configuration parameters for this validation case.

    Args:
        quick_test: If True, use reduced parameters for faster testing.

    Returns:
        Dictionary with all physics and numerical parameters.
    """
    if quick_test:
        return {
            "nr": 32,
            "nz": 32,
            "t_end": 0.05,  # Very short for quick test
            "dt": 1e-4,
            "v0": 1.0,
            "B0": 1.0,
        }

    return {
        "nr": 256,
        "nz": 256,
        "t_end": 0.5,
        "dt": 1e-4,
        "v0": 1.0,
        "B0": 1.0,
    }


def run_simulation(cfg: dict) -> tuple:
    """Run the Orszag-Tang vortex simulation.

    Args:
        cfg: Configuration dictionary from setup_configuration().

    Returns:
        Tuple of (final_state, geometry, history) where history is a dict
        containing time traces and snapshots.
    """
    # Build configuration with specified parameters
    config = OrszagTangConfiguration(
        nx=cfg["nr"],
        nz=cfg["nz"],
    )

    # Create geometry and initial state
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()

    # Create solver - explicit for MHD turbulence
    solver = Solver.create({"type": "euler"})

    # Time stepping parameters
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)

    # History capture - record at intervals
    output_interval = max(1, n_steps // 100)  # ~100 snapshots
    history = {
        "times": [],
        "total_energy": [],
        "magnetic_energy": [],
        "kinetic_energy": [],
        "thermal_energy": [],
        "max_current_density": [],
        "J_snapshots": [],
        "snapshot_times": [],
    }

    # Record initial state
    energies = compute_energies(state, geometry)
    J_mag = compute_current_magnitude(state, geometry)
    history["times"].append(0.0)
    history["total_energy"].append(float(energies["total"]))
    history["magnetic_energy"].append(float(energies["magnetic"]))
    history["kinetic_energy"].append(float(energies["kinetic"]))
    history["thermal_energy"].append(float(energies["thermal"]))
    history["max_current_density"].append(float(jnp.max(J_mag)))
    history["J_snapshots"].append(J_mag)
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
            energies = compute_energies(state, geometry)
            J_mag = compute_current_magnitude(state, geometry)

            history["times"].append(current_time)
            history["total_energy"].append(float(energies["total"]))
            history["magnetic_energy"].append(float(energies["magnetic"]))
            history["kinetic_energy"].append(float(energies["kinetic"]))
            history["thermal_energy"].append(float(energies["thermal"]))
            history["max_current_density"].append(float(jnp.max(J_mag)))

            # Store J snapshots at coarser intervals (every 10th recording)
            if len(history["times"]) % 10 == 0:
                history["J_snapshots"].append(J_mag)
                history["snapshot_times"].append(current_time)

        # Progress indicator
        progress_interval = max(1, n_steps // 10)
        if (step_idx + 1) % progress_interval == 0:
            print(f"    Step {step_idx + 1}/{n_steps} (t={state.time:.4f})")

    # Convert lists to arrays for easier processing
    history["times"] = jnp.array(history["times"])
    history["total_energy"] = jnp.array(history["total_energy"])
    history["magnetic_energy"] = jnp.array(history["magnetic_energy"])
    history["kinetic_energy"] = jnp.array(history["kinetic_energy"])
    history["thermal_energy"] = jnp.array(history["thermal_energy"])
    history["max_current_density"] = jnp.array(history["max_current_density"])

    return state, geometry, history


def compute_energies(state, geometry) -> dict:
    """Compute energy components (magnetic, kinetic, thermal).

    Args:
        state: Current simulation state.
        geometry: Computational geometry.

    Returns:
        Dictionary with 'magnetic', 'kinetic', 'thermal', and 'total' energies.
    """
    dr = (geometry.r_max - geometry.r_min) / geometry.nr
    dz = (geometry.z_max - geometry.z_min) / geometry.nz
    r = geometry.r_grid

    # Magnetic energy: |B|^2 / 2
    B_mag_sq = jnp.sum(state.B**2, axis=-1)
    E_mag = 0.5 * jnp.sum(B_mag_sq * r * dr * dz)

    # Kinetic energy: rho * |v|^2 / 2
    v_mag_sq = jnp.sum(state.v**2, axis=-1)
    E_kin = 0.5 * jnp.sum(state.n * v_mag_sq * r * dr * dz)

    # Thermal energy: p / (gamma - 1)
    gamma = 5.0 / 3.0
    E_thermal = jnp.sum(state.p / (gamma - 1.0) * r * dr * dz)

    return {
        "magnetic": E_mag,
        "kinetic": E_kin,
        "thermal": E_thermal,
        "total": E_mag + E_kin + E_thermal,
    }


def compute_current_magnitude(state, geometry) -> jnp.ndarray:
    """Compute magnitude of current density J = curl(B).

    Uses simple centered finite differences.

    Args:
        state: Current simulation state.
        geometry: Computational geometry.

    Returns:
        2D array of |J| magnitude.
    """
    dr = (geometry.r_max - geometry.r_min) / geometry.nr
    dz = (geometry.z_max - geometry.z_min) / geometry.nz

    Br = state.B[:, :, 0]
    Btheta = state.B[:, :, 1]
    Bz = state.B[:, :, 2]

    # J_r = (1/r) * dBz/dtheta - dBtheta/dz (axisymmetric: dBz/dtheta = 0)
    dBtheta_dz = jnp.gradient(Btheta, dz, axis=1)
    J_r = -dBtheta_dz

    # J_theta = dBr/dz - dBz/dr
    dBr_dz = jnp.gradient(Br, dz, axis=1)
    dBz_dr = jnp.gradient(Bz, dr, axis=0)
    J_theta = dBr_dz - dBz_dr

    # J_z = (1/r) * d(r*Btheta)/dr (axisymmetric)
    r = geometry.r_grid
    r_safe = jnp.maximum(r, 1e-10)  # Avoid division by zero
    rBtheta = r * Btheta
    drBtheta_dr = jnp.gradient(rBtheta, dr, axis=0)
    J_z = drBtheta_dr / r_safe

    # Total magnitude
    J_mag = jnp.sqrt(J_r**2 + J_theta**2 + J_z**2)
    return J_mag


# Acceptance criteria for Orszag-Tang vortex validation
ACCEPTANCE = {
    "no_numerical_instability": {
        "description": "No NaN or Inf values in solution",
    },
    "energy_conservation": {
        "threshold": 0.01,  # 1% drift allowed
        "description": "Total energy conserved within 1%",
    },
    "peak_J_timing": {
        "expected": 0.48,
        "tolerance": 0.10,  # 10% tolerance
        "description": "Peak current density timing within 10%",
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


def check_energy_conservation(history, threshold: float) -> tuple:
    """Check if total energy is conserved within threshold.

    Args:
        history: History dictionary containing energy time trace.
        threshold: Maximum allowed relative change (e.g., 0.01 for 1%).

    Returns:
        Tuple of (passed, relative_drift) where passed is bool and
        relative_drift is the maximum relative change from initial.
    """
    energy = history["total_energy"]
    initial_energy = energy[0]

    if jnp.abs(initial_energy) < 1e-10:
        # Avoid division by near-zero
        return True, 0.0

    relative_change = jnp.abs((energy - initial_energy) / initial_energy)
    max_drift = float(jnp.max(relative_change))

    passed = max_drift < threshold
    return passed, max_drift


def check_peak_J_timing(history, expected: float, tolerance: float) -> tuple:
    """Check if time of peak current density matches expected value.

    Args:
        history: History dictionary containing max J time trace.
        expected: Expected time to peak J.
        tolerance: Relative tolerance (e.g., 0.10 for 10%).

    Returns:
        Tuple of (passed, measured_time).
    """
    peak_idx = int(jnp.argmax(history["max_current_density"]))
    time_to_peak = float(history["times"][peak_idx])

    if expected > 0:
        relative_error = abs(time_to_peak - expected) / expected
        passed = relative_error < tolerance
    else:
        passed = True

    return passed, time_to_peak


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


def plot_energy_partition(history, figsize=(10, 5)):
    """Create plot showing energy partition over time.

    Args:
        history: History dictionary containing energy traces.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(history["times"], history["magnetic_energy"], 'b-', label='Magnetic', linewidth=1.5)
    ax.plot(history["times"], history["kinetic_energy"], 'r-', label='Kinetic', linewidth=1.5)
    ax.plot(history["times"], history["thermal_energy"], 'g-', label='Thermal', linewidth=1.5)
    ax.plot(history["times"], history["total_energy"], 'k--', label='Total', linewidth=2)

    ax.set_xlabel('Time [Alfven times]')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Partition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_contour(field, geometry, title: str, figsize=(8, 6)):
    """Create a contour plot of a 2D field.

    Args:
        field: 2D array (nr x nz).
        geometry: Computational geometry.
        title: Plot title.
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    r_range = jnp.linspace(geometry.r_min, geometry.r_max, geometry.nr)
    z_range = jnp.linspace(geometry.z_min, geometry.z_max, geometry.nz)

    # Transpose for correct orientation (r on x-axis, z on y-axis)
    c = ax.contourf(r_range, z_range, field.T, levels=50, cmap='hot')
    plt.colorbar(c, ax=ax)

    ax.set_xlabel('r')
    ax.set_ylabel('z')
    ax.set_title(title)
    ax.set_aspect('equal')

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

    # 2. Energy conservation
    # In quick test mode, use relaxed threshold since simulation is too short
    energy_threshold = ACCEPTANCE["energy_conservation"]["threshold"]
    if quick_test:
        energy_threshold = 0.1  # Relaxed to 10% for quick test
    energy_pass, energy_drift = check_energy_conservation(history, energy_threshold)
    print(f"  Energy conservation: {'PASS' if energy_pass else 'FAIL'} "
          f"(drift: {energy_drift:.4g}, threshold: {energy_threshold})")

    # 3. Peak J timing (optional in quick test)
    if quick_test:
        peak_J_pass = True
        time_to_peak_J = float(history["times"][int(jnp.argmax(history["max_current_density"]))])
        print("  (Skipping peak J timing check in quick test mode)")
    else:
        expected_time = ACCEPTANCE["peak_J_timing"]["expected"]
        time_tol = ACCEPTANCE["peak_J_timing"]["tolerance"]
        peak_J_pass, time_to_peak_J = check_peak_J_timing(history, expected_time, time_tol)
        print(f"  Peak J timing: {'PASS' if peak_J_pass else 'FAIL'} "
              f"(measured: {time_to_peak_J:.4f}, expected: {expected_time} +/- {time_tol*100}%)")

    print()

    overall_pass = stability_pass and energy_pass and (quick_test or peak_J_pass)

    # Build metrics dictionary for report
    metrics = {
        "numerical_stability": {
            "value": "stable" if stability_pass else "unstable",
            "passed": stability_pass,
            "description": ACCEPTANCE["no_numerical_instability"]["description"],
        },
        "energy_conservation": {
            "value": energy_drift,
            "threshold": energy_threshold,
            "passed": energy_pass,
            "description": ACCEPTANCE["energy_conservation"]["description"],
        },
        "peak_J_timing": {
            "value": time_to_peak_J,
            "expected": ACCEPTANCE["peak_J_timing"]["expected"],
            "threshold": f"+/- {ACCEPTANCE['peak_J_timing']['tolerance']*100}%",
            "passed": peak_J_pass if not quick_test else True,
            "description": ACCEPTANCE["peak_J_timing"]["description"],
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

    # Add total energy evolution plot
    initial_energy = history["total_energy"][0]
    if jnp.abs(initial_energy) > 1e-10:
        normalized_energy = history["total_energy"] / initial_energy
        fig_energy = plot_time_trace(
            times=history["times"],
            values=normalized_energy,
            title="Normalized Total Energy (relative to initial)",
            ylabel="Energy / Initial Energy",
        )
        report.add_plot(fig_energy, name="energy_normalized")
        plt.close(fig_energy)

    # Add energy partition plot
    fig_partition = plot_energy_partition(history)
    report.add_plot(fig_partition, name="energy_partition")
    plt.close(fig_partition)

    # Add max current density evolution
    fig_J = plot_time_trace(
        times=history["times"],
        values=history["max_current_density"],
        title="Maximum Current Density",
        ylabel="max |J|",
    )
    report.add_plot(fig_J, name="max_current")
    plt.close(fig_J)

    # Add final J contour if we have snapshots
    if len(history["J_snapshots"]) > 0:
        fig_J_contour = plot_contour(
            history["J_snapshots"][-1],
            geometry,
            f"|J| at t={history['snapshot_times'][-1]:.4f}",
        )
        report.add_plot(fig_J_contour, name="J_final")
        plt.close(fig_J_contour)

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
        if not energy_pass:
            print(f"  - Energy drift {energy_drift:.4g} exceeds threshold {energy_threshold}")
        if not quick_test and not peak_J_pass:
            print(f"  - Peak J timing {time_to_peak_J:.4f} outside tolerance")

    return overall_pass


if __name__ == "__main__":
    # Check for --quick flag
    quick = "--quick" in sys.argv
    success = main(quick_test=quick)
    sys.exit(0 if success else 1)
