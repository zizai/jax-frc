"""Translation validation: analytic benchmarks and model comparison.

This module provides validation infrastructure for FRC translation physics,
including:
- Tier 1: Analytic benchmarks (rigid FRC in field gradient)
- Tier 2: Model comparison (same IC, different physics models)
- Tier 3: Staged acceleration (programmable coil timing)
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import jax.numpy as jnp
from jax_frc.fields import CoilField, MirrorCoil, ThetaPinchArray


@dataclass
class AnalyticTrajectory:
    """Analytic trajectory for mirror force acceleration.

    For an FRC with magnetic moment mu in a field gradient dB/dz,
    the force is F = -mu * dB/dz, giving acceleration a = F/m.

    This assumes:
    - Constant field gradient (linear field)
    - Adiabatic invariance (mu conserved)
    - No flux loss or heating effects

    Args:
        magnetic_moment: Magnetic moment mu [A*m^2]
        frc_mass: Effective FRC mass [kg]
        field_gradient: Field gradient dB/dz [T/m]
        initial_position: Initial z position [m]
        initial_velocity: Initial z velocity [m/s]
    """
    magnetic_moment: float  # mu [A*m^2]
    frc_mass: float  # effective mass [kg]
    field_gradient: float  # dB/dz [T/m]
    initial_position: float  # z0 [m]
    initial_velocity: float  # v0 [m/s]

    @property
    def acceleration(self) -> float:
        """Compute constant acceleration from mirror force."""
        return -self.magnetic_moment * self.field_gradient / self.frc_mass

    def position(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute analytic position z(t).

        Uses kinematic equation: z = z0 + v0*t + 0.5*a*t^2

        Args:
            t: Time array [s]

        Returns:
            Position array [m]
        """
        a = self.acceleration
        return self.initial_position + self.initial_velocity * t + 0.5 * a * t**2

    def velocity(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute analytic velocity v(t).

        Uses kinematic equation: v = v0 + a*t

        Args:
            t: Time array [s]

        Returns:
            Velocity array [m/s]
        """
        a = self.acceleration
        return self.initial_velocity + a * t


@dataclass
class TranslationResult:
    """Result from a translation simulation.

    Stores time-history data from FRC translation, including
    position, velocity, flux conservation, and heating metrics.

    Args:
        times: Time points [s]
        positions: Centroid z position vs time [m]
        velocities: Centroid z velocity vs time [m/s]
        flux_max: Peak psi vs time (for flux conservation tracking)
        E_thermal: Thermal energy vs time [J] (for heating tracking)
    """
    times: jnp.ndarray
    positions: jnp.ndarray  # centroid z position vs time
    velocities: jnp.ndarray  # centroid z velocity vs time
    flux_max: jnp.ndarray  # peak psi vs time (flux conservation)
    E_thermal: jnp.ndarray  # thermal energy vs time (heating)

    def flux_loss_fraction(self) -> float:
        """Compute fractional flux loss over simulation.

        Returns:
            (psi_initial - psi_final) / psi_initial
        """
        psi_0 = float(self.flux_max[0])
        psi_f = float(self.flux_max[-1])
        if psi_0 == 0:
            return 0.0
        return (psi_0 - psi_f) / psi_0

    def heating_fraction(self) -> float:
        """Compute fractional heating over simulation.

        Returns:
            (E_final - E_initial) / E_initial
        """
        E_0 = float(self.E_thermal[0])
        E_f = float(self.E_thermal[-1])
        if E_0 == 0:
            return 0.0
        return (E_f - E_0) / E_0


@dataclass
class TranslationBenchmark:
    """Benchmark case for FRC translation validation.

    Defines a validation case with coil field configuration and
    optional analytic trajectory for comparison.

    Args:
        name: Benchmark identifier
        description: Human-readable description
        coil_field: External field source (CoilField protocol)
        analytic_trajectory: Optional analytic solution for comparison
    """
    name: str
    description: str
    coil_field: CoilField
    analytic_trajectory: Optional[AnalyticTrajectory] = None

    def compute_error(self, result: TranslationResult) -> dict:
        """Compute error metrics vs analytic trajectory.

        Args:
            result: Simulation result to compare

        Returns:
            Dictionary with error metrics:
            - max_position_error: Maximum absolute position error [m]
            - max_velocity_error: Maximum absolute velocity error [m/s]
            - rms_position_error: RMS position error [m]
            - rms_velocity_error: RMS velocity error [m/s]

            Empty dict if no analytic trajectory available.
        """
        if self.analytic_trajectory is None:
            return {}

        z_analytic = self.analytic_trajectory.position(result.times)
        v_analytic = self.analytic_trajectory.velocity(result.times)

        position_error = jnp.abs(result.positions - z_analytic)
        velocity_error = jnp.abs(result.velocities - v_analytic)

        return {
            "max_position_error": float(jnp.max(position_error)),
            "max_velocity_error": float(jnp.max(velocity_error)),
            "rms_position_error": float(jnp.sqrt(jnp.mean(position_error**2))),
            "rms_velocity_error": float(jnp.sqrt(jnp.mean(velocity_error**2))),
        }


@dataclass
class ModelComparisonResult:
    """Result from comparing multiple models on the same benchmark.

    Tier 2 validation: run same initial conditions through different
    physics models and compare trajectories, flux loss, and heating.

    Args:
        benchmark: The benchmark configuration used
        model_names: List of model names compared
        results: Dict mapping model name to TranslationResult
    """
    benchmark: TranslationBenchmark
    model_names: List[str]
    results: dict  # model_name -> TranslationResult

    def position_divergence(self) -> dict:
        """Compute pairwise position divergence between models.

        Returns:
            Dict mapping model pair to max position difference.
        """
        divergence = {}
        for i, name_i in enumerate(self.model_names):
            for name_j in self.model_names[i+1:]:
                result_i = self.results[name_i]
                result_j = self.results[name_j]
                max_diff = float(jnp.max(jnp.abs(
                    result_i.positions - result_j.positions
                )))
                divergence[f"{name_i}_vs_{name_j}"] = max_diff
        return divergence

    def flux_loss_comparison(self) -> dict:
        """Compare flux loss across models.

        Returns:
            Dict mapping model name to flux loss fraction.
        """
        return {
            name: self.results[name].flux_loss_fraction()
            for name in self.model_names
        }

    def heating_comparison(self) -> dict:
        """Compare heating across models.

        Returns:
            Dict mapping model name to heating fraction.
        """
        return {
            name: self.results[name].heating_fraction()
            for name in self.model_names
        }


def compute_field_gradient_at_point(
    coil_field: CoilField,
    r: float,
    z: float,
    t: float = 0.0,
    dz: float = 1e-4,
) -> float:
    """Compute axial field gradient dB_z/dz at a point.

    Uses central finite difference for numerical gradient.

    Args:
        coil_field: Field source
        r: Radial position [m]
        z: Axial position [m]
        t: Time [s]
        dz: Finite difference step [m]

    Returns:
        dB_z/dz at (r, z, t) [T/m]
    """
    r_arr = jnp.array([r])
    z_plus = jnp.array([z + dz])
    z_minus = jnp.array([z - dz])

    _, Bz_plus = coil_field.B_field(r_arr, z_plus, t)
    _, Bz_minus = coil_field.B_field(r_arr, z_minus, t)

    return float((Bz_plus[0] - Bz_minus[0]) / (2 * dz))


def create_mirror_push_benchmark(
    coil_separation: float = 2.0,
    coil_radius: float = 0.5,
    coil_current: float = 10000.0,
    frc_magnetic_moment: float = 1e-3,
    frc_mass: float = 1e-6,
    initial_offset: float = 0.1,
) -> TranslationBenchmark:
    """Create mirror push benchmark with two mirror coils.

    Two coils at +/- coil_separation/2 create a field minimum at center.
    An FRC placed slightly off-center will be pushed by the mirror force.

    Note: At exact center (z=0), the gradient is zero by symmetry.
    The FRC must start with a small offset to experience net force.

    Args:
        coil_separation: Distance between coil centers [m]
        coil_radius: Radius of each mirror coil [m]
        coil_current: Current in each coil [A]
        frc_magnetic_moment: FRC magnetic moment [A*m^2]
        frc_mass: FRC effective mass [kg]
        initial_offset: Initial z offset from center [m]

    Returns:
        TranslationBenchmark configured for mirror push test.
    """
    # Create two mirror coils with same polarity
    coil_positions = jnp.array([-coil_separation/2, coil_separation/2])

    coil_array = ThetaPinchArray(
        coil_positions=coil_positions,
        radii=coil_radius,
        currents=jnp.array([coil_current, coil_current]),
    )

    # Compute actual gradient at the initial position
    gradient = compute_field_gradient_at_point(
        coil_array, r=0.0, z=initial_offset
    )

    return TranslationBenchmark(
        name="mirror_push_analytic",
        description=f"Two mirror coils push FRC from offset z={initial_offset}m",
        coil_field=coil_array,
        analytic_trajectory=AnalyticTrajectory(
            magnetic_moment=frc_magnetic_moment,
            frc_mass=frc_mass,
            field_gradient=gradient,
            initial_position=initial_offset,
            initial_velocity=0.0,
        ),
    )


def create_uniform_gradient_benchmark(
    gradient: float = 0.1,
    background_field: float = 1.0,
    frc_magnetic_moment: float = 1e-3,
    frc_mass: float = 1e-6,
    initial_velocity: float = 0.0,
) -> TranslationBenchmark:
    """Create benchmark with idealized uniform gradient field.

    This uses a single mirror coil positioned to approximate a uniform
    gradient in the region of interest. Best for testing the basic
    mirror force physics.

    Args:
        gradient: Target field gradient dB/dz [T/m]
        background_field: Background axial field [T]
        frc_magnetic_moment: FRC magnetic moment [A*m^2]
        frc_mass: FRC effective mass [kg]
        initial_velocity: Initial z velocity [m/s]

    Returns:
        TranslationBenchmark for uniform gradient test.
    """
    # For a uniform gradient, we use a single coil far away
    # The FRC starts at z=0 where the gradient is approximately linear
    # This is a simplified model - real benchmarks would use actual coil configs

    # Use a solenoid with tapered current to create gradient
    # For now, use a mirror coil as placeholder
    coil = MirrorCoil(
        z_position=5.0,  # Far from test region
        radius=1.0,
        current=background_field * 2 * jnp.pi / (4e-7 * jnp.pi),  # Approximate
    )

    return TranslationBenchmark(
        name="uniform_gradient",
        description=f"Uniform gradient dB/dz = {gradient} T/m",
        coil_field=coil,
        analytic_trajectory=AnalyticTrajectory(
            magnetic_moment=frc_magnetic_moment,
            frc_mass=frc_mass,
            field_gradient=gradient,
            initial_position=0.0,
            initial_velocity=initial_velocity,
        ),
    )


def create_staged_acceleration_benchmark(
    n_stages: int = 3,
    stage_spacing: float = 1.0,
    coil_radius: float = 0.5,
    peak_current: float = 50000.0,
    timing_sequence: Optional[Callable[[float], jnp.ndarray]] = None,
) -> TranslationBenchmark:
    """Create staged acceleration benchmark with programmable coil timing.

    Tier 3 validation: FRC is accelerated through multiple stages
    with time-dependent coil currents.

    Args:
        n_stages: Number of acceleration stages
        stage_spacing: Distance between stage centers [m]
        coil_radius: Radius of stage coils [m]
        peak_current: Peak current per stage [A]
        timing_sequence: Function t -> currents array, or None for static

    Returns:
        TranslationBenchmark for staged acceleration.
    """
    # Position coils along z-axis
    coil_positions = jnp.linspace(0, (n_stages - 1) * stage_spacing, n_stages)

    # Default timing: all coils at peak current (static test)
    if timing_sequence is None:
        currents = jnp.full(n_stages, peak_current)
    else:
        currents = timing_sequence

    coil_array = ThetaPinchArray(
        coil_positions=coil_positions,
        radii=coil_radius,
        currents=currents,
    )

    return TranslationBenchmark(
        name=f"staged_acceleration_{n_stages}",
        description=f"{n_stages}-stage acceleration with spacing {stage_spacing}m",
        coil_field=coil_array,
        analytic_trajectory=None,  # No simple analytic solution for staged
    )


def traveling_wave_timing(
    n_coils: int,
    wave_speed: float,
    coil_spacing: float,
    peak_current: float,
    rise_time: float = 1e-6,
) -> Callable[[float], jnp.ndarray]:
    """Create traveling wave timing function for staged acceleration.

    Each coil fires in sequence as the "wave" passes, creating
    a translating magnetic well that drags the FRC along.

    Args:
        n_coils: Number of coils
        wave_speed: Speed of traveling wave [m/s]
        coil_spacing: Distance between coils [m]
        peak_current: Peak current per coil [A]
        rise_time: Current rise time [s]

    Returns:
        Function t -> currents array for use with ThetaPinchArray.
    """
    def timing(t: float) -> jnp.ndarray:
        # Wave position at time t
        wave_pos = wave_speed * t

        # Coil positions
        coil_pos = jnp.arange(n_coils) * coil_spacing

        # Each coil fires when wave reaches it
        # Use smooth ramp-up
        delay = coil_pos / wave_speed
        relative_time = t - delay

        # Ramp profile: smooth rise to peak
        ramp = jnp.clip(relative_time / rise_time, 0.0, 1.0)
        currents = peak_current * ramp

        return currents

    return timing
