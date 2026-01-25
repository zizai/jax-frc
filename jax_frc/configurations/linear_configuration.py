# jax_frc/configurations/linear_configuration.py
"""LinearConfiguration for sequential multi-phase simulations.

This module replaces jax_frc/scenarios/scenario.py. LinearConfiguration
extends AbstractConfiguration and includes execution logic directly -
the configuration IS the runnable simulation.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging

from jax_frc.configurations.base import AbstractConfiguration
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase, PhaseResult, PHASE_REGISTRY
from jax_frc.scenarios.transitions import transition_from_spec
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers.base import Solver
from jax_frc.solvers.explicit import RK4Solver

if TYPE_CHECKING:
    from jax_frc.diagnostics.probes import Probe

log = logging.getLogger(__name__)


@dataclass
class TransitionSpec:
    """Declarative specification for a phase transition.

    Used to create Transition objects without direct instantiation,
    enabling serialization and configuration-driven phase definitions.

    Attributes:
        type: Transition type ("timeout", "separation_below", "temperature_above",
              "flux_below", "velocity_below", "any_of", "all_of", "condition")
        value: Threshold or timeout value (for simple transitions)
        children: List of child TransitionSpecs (for composite "any_of"/"all_of")
    """

    type: str
    value: Any = None
    children: Optional[List["TransitionSpec"]] = None


@dataclass
class PhaseSpec:
    """Declarative specification for a simulation phase.

    Attributes:
        name: Human-readable phase name
        transition: Specification for when the phase completes
        phase_class: Name of the Phase class to instantiate (from PHASE_REGISTRY)
        config: Configuration dict passed to phase constructor and setup()
    """

    name: str
    transition: TransitionSpec
    phase_class: str = "Phase"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationResult:
    """Result of running a LinearConfiguration.

    Replaces ScenarioResult - contains all phase results and overall outcome.

    Attributes:
        name: Configuration/scenario identifier
        phase_results: Results from each executed phase in order
        total_time: Total simulation time across all phases
        success: True if all phases completed without error
    """

    name: str
    phase_results: List[PhaseResult]
    total_time: float
    success: bool

    @property
    def final_state(self) -> State:
        """Get final state from last phase."""
        return self.phase_results[-1].final_state


@dataclass
class LinearConfiguration(AbstractConfiguration):
    """Configuration for sequential multi-phase simulations.

    Replaces Scenario - this is the primary way to define and run
    multi-phase FRC simulations. Extends AbstractConfiguration with
    execution logic built in.

    Subclasses must implement:
        - build_geometry() - define computational domain
        - build_initial_state(geometry) - define starting conditions
        - build_model() - define physics equations
        - build_boundary_conditions() - define BCs
        - build_phase_specs() - define phases declaratively

    Optional overrides:
        - build_solver() - customize time integrator (default: RK4)

    Example usage::

        @dataclass
        class FRCMergingConfiguration(LinearConfiguration):
            name: str = "frc_merging"
            separation: float = 1.0
            dt: float = 1e-6

            def build_geometry(self) -> Geometry:
                return Geometry(nr=64, nz=256, ...)

            def build_initial_state(self, geometry) -> State:
                return create_initial_frc(geometry)

            def build_model(self) -> PhysicsModel:
                return ExtendedMHD(...)

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="merging",
                        transition=TransitionSpec(type="timeout", value=1e-3),
                        phase_class="MergingPhase",
                        config={"separation": self.separation},
                    ),
                ]

        # Usage
        config = FRCMergingConfiguration(separation=0.8)
        result = config.run()
        print(result.final_state)
    """

    # Simulation parameters
    dt: float = 1e-6
    output_interval: int = 100

    # Phase configuration overrides (keyed by phase name)
    phase_config: Dict[str, dict] = field(default_factory=dict)

    # Diagnostics (runtime, not serialized)
    diagnostics: List["Probe"] = field(default_factory=list)

    @abstractmethod
    def build_phase_specs(self) -> List[PhaseSpec]:
        """Define simulation phases declaratively.

        Returns:
            List of PhaseSpec objects defining the phases in order
        """
        ...

    def build_solver(self) -> Solver:
        """Build time integrator. Override for custom solver.

        Returns:
            Solver instance (default: RK4Solver)
        """
        return RK4Solver()

    def build_phases(self, geometry: Geometry) -> List[Phase]:
        """Convert PhaseSpecs to Phase objects.

        Args:
            geometry: Geometry for transitions that need spatial info

        Returns:
            List of Phase instances ready to execute
        """
        specs = self.build_phase_specs()
        phases = []
        for spec in specs:
            phase_cls = PHASE_REGISTRY.get(spec.phase_class)
            if phase_cls is None:
                raise ValueError(f"Unknown phase class: {spec.phase_class}. "
                               f"Available: {list(PHASE_REGISTRY.keys())}")

            transition = transition_from_spec(spec.transition, geometry)

            # Merge spec config with any runtime phase_config overrides
            config = {**spec.config, **self.phase_config.get(spec.name, {})}

            # Create phase with name, transition, and any extra config
            phase = phase_cls(name=spec.name, transition=transition, **config)
            phases.append(phase)

        return phases

    def run(self) -> ConfigurationResult:
        """Execute all phases in sequence.

        Returns:
            ConfigurationResult with results from all phases
        """
        # Build all components
        geometry = self.build_geometry()
        state = self.build_initial_state(geometry)
        phases = self.build_phases(geometry)
        solver = self.build_solver()
        model = self.build_model()

        phase_results = []

        for phase in phases:
            log.info(f"Starting phase: {phase.name}")

            # Get phase-specific config
            phase_config = self.phase_config.get(phase.name, {})

            # Run the phase
            result = self._run_phase(phase, state, geometry, solver, model, phase_config)
            phase_results.append(result)

            # Pass state to next phase
            state = result.final_state

            if result.termination == "timeout":
                log.warning(f"Phase {phase.name} ended by timeout")
            elif result.termination == "error":
                log.error(f"Phase {phase.name} ended with error")
                break

        total_time = sum(r.end_time - r.start_time for r in phase_results)
        success = all(r.termination != "error" for r in phase_results)

        return ConfigurationResult(
            name=self.name,
            phase_results=phase_results,
            total_time=total_time,
            success=success,
        )

    def _run_phase(
        self,
        phase: Phase,
        state: State,
        geometry: Geometry,
        solver: Solver,
        model: PhysicsModel,
        config: dict,
    ) -> PhaseResult:
        """Run a single phase to completion.

        Args:
            phase: Phase to run
            state: Initial state for this phase
            geometry: Computational geometry
            solver: Time integrator
            model: Physics model
            config: Phase configuration

        Returns:
            PhaseResult with final state and diagnostics
        """
        # Setup phase
        start_time = float(state.time) if state else 0.0
        state = phase.setup(state, geometry, config)
        initial_state = state

        t = start_time
        termination = "condition_met"

        # Initialize history tracking
        history: Dict[str, List[float]] = {"time": []}
        for probe in self.diagnostics:
            history[probe.name] = []

        # Run until transition triggers
        while True:
            # Check for completion
            complete, reason = phase.is_complete(state, t)
            if complete:
                termination = reason
                break

            # Record diagnostics at intervals
            if state.step % self.output_interval == 0:
                history["time"].append(t)
                for probe in self.diagnostics:
                    history[probe.name].append(probe.measure(state, geometry))

            # Apply step hook (time-varying BCs, etc.)
            state = phase.step_hook(state, geometry, t)

            # Advance physics via solver
            state = solver.step(state, self.dt, model, geometry)
            t = float(state.time)

        # Cleanup
        state = phase.on_complete(state, geometry)

        return PhaseResult(
            name=phase.name,
            initial_state=initial_state,
            final_state=state,
            start_time=start_time,
            end_time=t,
            termination=termination,
            history=history,
        )
