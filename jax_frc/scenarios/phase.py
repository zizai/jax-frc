"""Phase base class for simulation stages."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.transitions import Transition


@dataclass
class PhaseResult:
    """Result of running a phase."""

    name: str
    initial_state: State
    final_state: State
    start_time: float
    end_time: float
    termination: str  # "condition_met", "timeout", "error"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Phase:
    """Base class for simulation phases.

    A phase represents one stage of an FRC experiment (formation, merging, etc.).
    Subclasses override setup, step_hook, and on_complete for phase-specific behavior.

    Attributes:
        name: Human-readable phase name
        transition: Condition for phase completion
    """

    name: str
    transition: Transition

    def setup(self, state: State, geometry: Geometry, config: dict) -> State:
        """Called once when phase begins.

        Override to modify initial state, configure boundary conditions, etc.

        Args:
            state: State from previous phase (or initial state)
            geometry: Computational geometry
            config: Phase-specific configuration

        Returns:
            Modified state to begin phase with
        """
        return state

    def step_hook(self, state: State, geometry: Geometry, t: float) -> State:
        """Called each timestep during phase.

        Override for time-varying boundary conditions, source terms, etc.

        Args:
            state: Current state
            geometry: Computational geometry
            t: Current simulation time

        Returns:
            Modified state (or unchanged if no per-step modifications needed)
        """
        return state

    def is_complete(self, state: State, t: float) -> tuple[bool, str]:
        """Check if phase should end.

        Delegates to transition.evaluate(). Override for custom logic.

        Args:
            state: Current state
            t: Current simulation time

        Returns:
            (triggered, reason) tuple
        """
        return self.transition.evaluate(state, t)

    def on_complete(self, state: State, geometry: Geometry) -> State:
        """Called once when phase ends.

        Override for cleanup, state preparation for next phase, etc.

        Args:
            state: Final state of this phase
            geometry: Computational geometry

        Returns:
            State to pass to next phase
        """
        return state
