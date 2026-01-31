"""Simulation orchestrator with builder pattern."""
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from jax_frc.simulation.geometry import Geometry
    from jax_frc.simulation.state import State
    from jax_frc.models.base import PhysicsModel
    from jax_frc.solvers.base import Solver


@dataclass
class Simulation:
    """Main simulation orchestrator."""
    geometry: "Geometry"
    model: "PhysicsModel"
    solver: "Solver"
    state: "State"
    phases: List = field(default_factory=list)
    callbacks: List[Callable] = field(default_factory=list)
    
    @classmethod
    def builder(cls) -> "SimulationBuilder":
        return SimulationBuilder()
    
    def step(self) -> "State":
        """Advance simulation by one timestep."""
        self.state = self.solver.step(self.state, self.model, self.geometry)
        return self.state
    
    def run(self, t_end: float) -> "State":
        """Run simulation until t_end."""
        while self.state.time < t_end:
            self.step()
            for cb in self.callbacks:
                cb(self.state)
        return self.state


class SimulationBuilder:
    """Fluent builder for Simulation."""
    
    def __init__(self):
        self._geometry = None
        self._model = None
        self._solver = None
        self._state = None
        self._phases = []
        self._callbacks = []
    
    def geometry(self, g) -> "SimulationBuilder":
        self._geometry = g
        return self
    
    def model(self, m) -> "SimulationBuilder":
        self._model = m
        return self
    
    def solver(self, s) -> "SimulationBuilder":
        self._solver = s
        return self
    
    def initial_state(self, s) -> "SimulationBuilder":
        self._state = s
        return self
    
    def phases(self, p) -> "SimulationBuilder":
        self._phases = p
        return self
    
    def callbacks(self, c) -> "SimulationBuilder":
        self._callbacks = c
        return self
    
    def build(self) -> Simulation:
        if self._geometry is None:
            raise ValueError("geometry is required")
        if self._model is None:
            raise ValueError("model is required")
        if self._solver is None:
            raise ValueError("solver is required")
        if self._state is None:
            raise ValueError("initial_state is required")
        
        return Simulation(
            geometry=self._geometry,
            model=self._model,
            solver=self._solver,
            state=self._state,
            phases=self._phases,
            callbacks=self._callbacks,
        )
