# jax_frc/scenarios/scenario.py
"""Scenario runner for multi-phase simulations."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers.base import Solver
from jax_frc.diagnostics.probes import Probe

log = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a complete scenario."""

    name: str
    phase_results: List[PhaseResult]
    total_time: float
    success: bool

    @property
    def final_state(self) -> State:
        """Get final state from last phase."""
        return self.phase_results[-1].final_state


@dataclass
class Scenario:
    """Orchestrates multi-phase FRC simulations.

    Runs phases in sequence, passing state between them.
    Each phase runs until its transition triggers.

    Attributes:
        name: Scenario identifier
        phases: Ordered list of phases to run
        geometry: Computational geometry
        initial_state: Starting state (or None if first phase creates it)
        physics_model: Physics model for evolution equations
        solver: Time integrator
        dt: Timestep for simulation
        config: Optional per-phase configuration overrides
    """

    name: str
    phases: List[Phase]
    geometry: Geometry
    initial_state: Optional[State]
    physics_model: PhysicsModel
    solver: Solver
    dt: float
    config: Dict[str, dict] = field(default_factory=dict)
    diagnostics: List[Probe] = field(default_factory=list)
    output_interval: int = 100

    def run(self) -> ScenarioResult:
        """Run all phases in sequence.

        Returns:
            ScenarioResult with results from all phases
        """
        state = self.initial_state
        phase_results = []

        for phase in self.phases:
            log.info(f"Starting phase: {phase.name}")

            # Get phase-specific config
            phase_config = self.config.get(phase.name, {})

            # Run the phase
            result = self._run_phase(phase, state, phase_config)
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

        return ScenarioResult(
            name=self.name,
            phase_results=phase_results,
            total_time=total_time,
            success=success,
        )

    def _run_phase(self, phase: Phase, state: State, config: dict) -> PhaseResult:
        """Run a single phase to completion.

        Args:
            phase: Phase to run
            state: Initial state for this phase
            config: Phase configuration

        Returns:
            PhaseResult with final state and diagnostics
        """
        # Setup phase
        start_time = float(state.time) if state else 0.0
        state = phase.setup(state, self.geometry, config)
        initial_state = state

        t = start_time
        termination = "condition_met"

        # Run until transition triggers
        while True:
            # Check for completion
            complete, reason = phase.is_complete(state, t)
            if complete:
                termination = reason
                break

            # Apply step hook (time-varying BCs, etc.)
            state = phase.step_hook(state, self.geometry, t)

            # Advance physics via solver
            state = self.solver.step(state, self.dt, self.physics_model, self.geometry)
            t = float(state.time)

        # Cleanup
        state = phase.on_complete(state, self.geometry)

        return PhaseResult(
            name=phase.name,
            initial_state=initial_state,
            final_state=state,
            start_time=start_time,
            end_time=t,
            termination=termination,
        )
