"""Abstract base class for time integrators."""

from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

class Solver(ABC):
    """Base class for time integration solvers."""

    @abstractmethod
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep."""
        pass

    @classmethod
    def create(cls, config: dict) -> "Solver":
        """Factory method to create solver from config."""
        solver_type = config.get("type", "euler")
        if solver_type == "euler":
            from jax_frc.solvers.explicit import EulerSolver
            return EulerSolver()
        elif solver_type == "rk4":
            from jax_frc.solvers.explicit import RK4Solver
            return RK4Solver()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
