"""Belova FRC merging comparison: resistive MHD vs hybrid kinetic.

This module provides a framework for comparing resistive MHD and hybrid
kinetic simulations of FRC merging, based on the methodology from:

Reference: Belova et al. 2025 - Hybrid Simulations of FRC Merging and Compression
           https://arxiv.org/abs/2501.03425

The comparison suite tracks key diagnostics including:
- Magnetic null separation vs time
- Reconnection rate vs time
- Energy partition (magnetic, kinetic, thermal)
- Merge time (when null separation -> 0)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.merging import MergingDiagnostics
from jax_frc.diagnostics.energy import EnergyDiagnostics


@dataclass
class MergingResult:
    """Result from a single merging simulation run.

    This data class stores the time series diagnostics from an FRC merging
    simulation, capturing the evolution of key physical quantities.

    Attributes:
        model_type: Type of model used ("resistive_mhd" or "hybrid_kinetic")
        times: Array of time points [s]
        null_separation: Distance between magnetic nulls vs time [m]
        reconnection_rate: Reconnection rate vs time [V/m or dimensionless]
        E_magnetic: Magnetic energy vs time [J]
        E_kinetic: Kinetic energy vs time [J]
        E_thermal: Thermal energy vs time [J]
        merge_time: Time when nulls merge (separation -> 0), or None if not merged
        final_state: Final simulation state
    """

    model_type: str
    times: Array
    null_separation: Array
    reconnection_rate: Array
    E_magnetic: Array
    E_kinetic: Array
    E_thermal: Array
    merge_time: Optional[float]
    final_state: State

    def __post_init__(self):
        """Validate model type."""
        valid_types = {"resistive_mhd", "hybrid_kinetic"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got '{self.model_type}'"
            )

    @property
    def E_total(self) -> Array:
        """Total energy vs time [J]."""
        return self.E_magnetic + self.E_kinetic + self.E_thermal

    @property
    def n_timesteps(self) -> int:
        """Number of time points recorded."""
        return len(self.times)

    def energy_fraction_at_time(self, t: float) -> Dict[str, float]:
        """Get energy fractions at a specific time.

        Args:
            t: Time to evaluate [s]

        Returns:
            Dictionary with f_magnetic, f_kinetic, f_thermal fractions
        """
        # Find closest time index
        idx = int(jnp.argmin(jnp.abs(self.times - t)))

        E_mag = float(self.E_magnetic[idx])
        E_kin = float(self.E_kinetic[idx])
        E_therm = float(self.E_thermal[idx])
        E_tot = E_mag + E_kin + E_therm

        if E_tot <= 0:
            return {"f_magnetic": 0.0, "f_kinetic": 0.0, "f_thermal": 0.0}

        return {
            "f_magnetic": E_mag / E_tot,
            "f_kinetic": E_kin / E_tot,
            "f_thermal": E_therm / E_tot,
        }


@dataclass
class ComparisonReport:
    """Comparison report between two models.

    This report facilitates comparison of resistive MHD and hybrid kinetic
    simulation results, providing methods to compute differences in key
    quantities.

    Attributes:
        mhd_result: Result from resistive MHD simulation
        hybrid_result: Result from hybrid kinetic simulation
    """

    mhd_result: MergingResult
    hybrid_result: MergingResult

    def __post_init__(self):
        """Validate that results are from different model types."""
        if self.mhd_result.model_type != "resistive_mhd":
            raise ValueError(
                f"mhd_result must have model_type='resistive_mhd', "
                f"got '{self.mhd_result.model_type}'"
            )
        if self.hybrid_result.model_type != "hybrid_kinetic":
            raise ValueError(
                f"hybrid_result must have model_type='hybrid_kinetic', "
                f"got '{self.hybrid_result.model_type}'"
            )

    def merge_time_difference(self) -> float:
        """Difference in merge times between models.

        Returns:
            hybrid_merge_time - mhd_merge_time [s], or nan if either didn't merge
        """
        if self.mhd_result.merge_time is not None and self.hybrid_result.merge_time is not None:
            return self.hybrid_result.merge_time - self.mhd_result.merge_time
        return float("nan")

    def merge_time_ratio(self) -> float:
        """Ratio of merge times (hybrid / MHD).

        Returns:
            hybrid_merge_time / mhd_merge_time, or nan if either didn't merge
        """
        if (
            self.mhd_result.merge_time is not None
            and self.hybrid_result.merge_time is not None
            and self.mhd_result.merge_time > 0
        ):
            return self.hybrid_result.merge_time / self.mhd_result.merge_time
        return float("nan")

    def energy_partition_at_merge(self) -> Dict[str, Dict[str, float]]:
        """Energy partition at merge time for both models.

        Returns:
            Dictionary with 'mhd' and 'hybrid' keys, each containing
            f_magnetic, f_kinetic, f_thermal fractions. If a model
            didn't merge, uses final time.
        """
        # Get merge time or final time for MHD
        if self.mhd_result.merge_time is not None:
            mhd_time = self.mhd_result.merge_time
        else:
            mhd_time = float(self.mhd_result.times[-1])

        # Get merge time or final time for hybrid
        if self.hybrid_result.merge_time is not None:
            hybrid_time = self.hybrid_result.merge_time
        else:
            hybrid_time = float(self.hybrid_result.times[-1])

        return {
            "mhd": self.mhd_result.energy_fraction_at_time(mhd_time),
            "hybrid": self.hybrid_result.energy_fraction_at_time(hybrid_time),
        }

    def max_reconnection_rate_ratio(self) -> float:
        """Ratio of maximum reconnection rates (hybrid / MHD).

        Returns:
            max(hybrid_reconnection_rate) / max(mhd_reconnection_rate)
        """
        mhd_max = float(jnp.max(jnp.abs(self.mhd_result.reconnection_rate)))
        hybrid_max = float(jnp.max(jnp.abs(self.hybrid_result.reconnection_rate)))

        if mhd_max > 0:
            return hybrid_max / mhd_max
        return float("nan")

    def summary(self) -> Dict[str, float]:
        """Generate summary metrics for the comparison.

        Returns:
            Dictionary of key comparison metrics
        """
        return {
            "mhd_merge_time": self.mhd_result.merge_time or float("nan"),
            "hybrid_merge_time": self.hybrid_result.merge_time or float("nan"),
            "merge_time_difference": self.merge_time_difference(),
            "merge_time_ratio": self.merge_time_ratio(),
            "max_reconnection_rate_ratio": self.max_reconnection_rate_ratio(),
        }


@dataclass
class BelovaComparisonSuite:
    """Suite for comparing resistive MHD vs hybrid kinetic FRC merging.

    This suite implements the comparison methodology from Belova et al. 2025,
    allowing systematic comparison of different physics models during FRC
    merging events.

    The default parameters are representative values for FRC merging experiments.
    Users should adjust these based on their specific simulation needs.

    Attributes:
        initial_separation: Initial distance between FRC centers [m]
        initial_velocity: Initial axial velocity of each FRC [m/s]
        frc_elongation: FRC elongation (length/diameter)
        nr: Number of radial grid points
        nz: Number of axial grid points
        r_min: Minimum radial coordinate [m]
        r_max: Maximum radial coordinate [m]
        z_min: Minimum axial coordinate [m]
        z_max: Maximum axial coordinate [m]
    """

    # Shared parameters from Belova paper (representative values)
    initial_separation: float = 2.0  # [m]
    initial_velocity: float = 0.0  # [m/s]
    frc_elongation: float = 3.0

    # Grid parameters
    nr: int = 64
    nz: int = 128
    r_min: float = 0.01  # Avoid r=0 singularity
    r_max: float = 0.5
    z_min: float = -2.0
    z_max: float = 2.0

    # Internal diagnostics instances
    _merging_diag: MergingDiagnostics = field(
        default_factory=MergingDiagnostics, repr=False
    )
    _energy_diag: EnergyDiagnostics = field(
        default_factory=EnergyDiagnostics, repr=False
    )

    def create_geometry(self) -> Geometry:
        """Create computational geometry from suite parameters.

        Returns:
            Geometry object for simulations
        """
        return Geometry(
            coord_system="cylindrical",
            nr=self.nr,
            nz=self.nz,
            r_min=self.r_min,
            r_max=self.r_max,
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def collect_diagnostics(
        self, state: State, geometry: Geometry
    ) -> Dict[str, float]:
        """Collect all diagnostics from a simulation state.

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Dictionary with all diagnostic values
        """
        merging = self._merging_diag.compute(state, geometry)
        energy = self._energy_diag.compute(state, geometry)

        return {
            "separation": merging["separation_dz"],
            "reconnection_rate": merging["reconnection_rate"],
            "E_magnetic": energy["E_magnetic"],
            "E_kinetic": energy["E_kinetic"],
            "E_thermal": energy["E_thermal"],
        }

    def detect_merge_time(
        self, times: Array, separations: Array, threshold: float = 0.01
    ) -> Optional[float]:
        """Detect when FRCs have merged based on null separation.

        Args:
            times: Array of time points
            separations: Array of null separations at each time
            threshold: Separation threshold below which FRCs are considered merged [m]

        Returns:
            Time of merge, or None if merge didn't occur
        """
        # Find first time when separation drops below threshold
        merged = separations < threshold
        if not jnp.any(merged):
            return None

        merge_idx = int(jnp.argmax(merged))
        return float(times[merge_idx])

    def run_resistive_mhd(
        self,
        t_end: float,
        dt: float,
        initial_state: Optional[State] = None,
        resistivity_eta: float = 1e-6,
    ) -> MergingResult:
        """Run resistive MHD simulation and collect diagnostics.

        This method creates and runs a resistive MHD simulation of FRC merging,
        collecting diagnostics at each output interval.

        Args:
            t_end: End time for simulation [s]
            dt: Time step [s]
            initial_state: Optional initial state (creates default if None)
            resistivity_eta: Resistivity value [Ohm*m]

        Returns:
            MergingResult with collected diagnostics

        Note:
            This is a stub implementation. Full implementation requires
            integrating with the simulation runner infrastructure.
        """
        # Import models here to avoid circular imports
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        geometry = self.create_geometry()

        # Create model
        resistivity = SpitzerResistivity(eta_0=resistivity_eta)
        model = ResistiveMHD(resistivity=resistivity)

        # Create initial state if not provided
        if initial_state is None:
            initial_state = State.zeros(self.nr, self.nz)

        # Placeholder for actual simulation loop
        # In production, this would use the simulation runner
        n_steps = int(t_end / dt)
        n_outputs = min(n_steps, 1000)  # Limit output points
        output_interval = max(1, n_steps // n_outputs)

        # Initialize storage
        times_list = []
        separation_list = []
        recon_rate_list = []
        E_mag_list = []
        E_kin_list = []
        E_therm_list = []

        state = initial_state

        # Collect initial diagnostics
        diag = self.collect_diagnostics(state, geometry)
        times_list.append(0.0)
        separation_list.append(diag["separation"])
        recon_rate_list.append(diag["reconnection_rate"])
        E_mag_list.append(diag["E_magnetic"])
        E_kin_list.append(diag["E_kinetic"])
        E_therm_list.append(diag["E_thermal"])

        # NOTE: Actual time stepping would go here
        # For now, we just return the initial state diagnostics
        # This is a framework stub - full integration requires
        # connecting to the simulation infrastructure

        # Convert to arrays
        times = jnp.array(times_list)
        null_separation = jnp.array(separation_list)
        reconnection_rate = jnp.array(recon_rate_list)
        E_magnetic = jnp.array(E_mag_list)
        E_kinetic = jnp.array(E_kin_list)
        E_thermal = jnp.array(E_therm_list)

        # Detect merge time
        merge_time = self.detect_merge_time(times, null_separation)

        return MergingResult(
            model_type="resistive_mhd",
            times=times,
            null_separation=null_separation,
            reconnection_rate=reconnection_rate,
            E_magnetic=E_magnetic,
            E_kinetic=E_kinetic,
            E_thermal=E_thermal,
            merge_time=merge_time,
            final_state=state,
        )

    def run_hybrid_kinetic(
        self,
        t_end: float,
        dt: float,
        initial_state: Optional[State] = None,
        n_particles: int = 10000,
        eta: float = 1e-6,
    ) -> MergingResult:
        """Run hybrid kinetic simulation and collect diagnostics.

        This method creates and runs a hybrid kinetic (PIC ions + fluid electrons)
        simulation of FRC merging, collecting diagnostics at each output interval.

        Args:
            t_end: End time for simulation [s]
            dt: Time step [s]
            initial_state: Optional initial state (creates default if None)
            n_particles: Number of particles for kinetic ions
            eta: Resistivity value [Ohm*m]

        Returns:
            MergingResult with collected diagnostics

        Note:
            This is a stub implementation. Full implementation requires
            integrating with the simulation runner infrastructure.
        """
        # Import models here to avoid circular imports
        from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium

        geometry = self.create_geometry()

        # Create model
        equilibrium = RigidRotorEquilibrium()
        model = HybridKinetic(equilibrium=equilibrium, eta=eta)

        # Create initial state if not provided
        if initial_state is None:
            initial_state = State.zeros(
                self.nr, self.nz, with_particles=True, n_particles=n_particles
            )

        # Placeholder for actual simulation loop
        n_steps = int(t_end / dt)
        n_outputs = min(n_steps, 1000)
        output_interval = max(1, n_steps // n_outputs)

        # Initialize storage
        times_list = []
        separation_list = []
        recon_rate_list = []
        E_mag_list = []
        E_kin_list = []
        E_therm_list = []

        state = initial_state

        # Collect initial diagnostics
        diag = self.collect_diagnostics(state, geometry)
        times_list.append(0.0)
        separation_list.append(diag["separation"])
        recon_rate_list.append(diag["reconnection_rate"])
        E_mag_list.append(diag["E_magnetic"])
        E_kin_list.append(diag["E_kinetic"])
        E_therm_list.append(diag["E_thermal"])

        # NOTE: Actual time stepping would go here

        # Convert to arrays
        times = jnp.array(times_list)
        null_separation = jnp.array(separation_list)
        reconnection_rate = jnp.array(recon_rate_list)
        E_magnetic = jnp.array(E_mag_list)
        E_kinetic = jnp.array(E_kin_list)
        E_thermal = jnp.array(E_therm_list)

        # Detect merge time
        merge_time = self.detect_merge_time(times, null_separation)

        return MergingResult(
            model_type="hybrid_kinetic",
            times=times,
            null_separation=null_separation,
            reconnection_rate=reconnection_rate,
            E_magnetic=E_magnetic,
            E_kinetic=E_kinetic,
            E_thermal=E_thermal,
            merge_time=merge_time,
            final_state=state,
        )

    def compare(
        self, mhd_result: MergingResult, hybrid_result: MergingResult
    ) -> ComparisonReport:
        """Generate comparison report from two simulation results.

        Args:
            mhd_result: Result from resistive MHD simulation
            hybrid_result: Result from hybrid kinetic simulation

        Returns:
            ComparisonReport with comparison metrics
        """
        return ComparisonReport(mhd_result=mhd_result, hybrid_result=hybrid_result)

    def run_comparison(
        self,
        t_end: float,
        dt: float,
        mhd_state: Optional[State] = None,
        hybrid_state: Optional[State] = None,
    ) -> ComparisonReport:
        """Run both simulations and generate comparison.

        Convenience method that runs both resistive MHD and hybrid kinetic
        simulations with the same parameters and generates a comparison report.

        Args:
            t_end: End time for simulations [s]
            dt: Time step [s]
            mhd_state: Optional initial state for MHD (creates default if None)
            hybrid_state: Optional initial state for hybrid (creates default if None)

        Returns:
            ComparisonReport comparing the two simulations
        """
        mhd_result = self.run_resistive_mhd(t_end, dt, initial_state=mhd_state)
        hybrid_result = self.run_hybrid_kinetic(t_end, dt, initial_state=hybrid_state)

        return self.compare(mhd_result, hybrid_result)
