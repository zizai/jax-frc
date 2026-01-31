"""Numerical recipe for bundling solver and constraint strategies.

DEPRECATED: NumericalRecipe functionality has been absorbed into Solver.
Use Solver.step() directly - it now handles timestep control and constraints.
"""
import warnings
from dataclasses import dataclass
from typing import Literal

from jax_frc.solvers.base import Solver
from jax_frc.solvers.time_controller import TimeController
from jax_frc.solvers.divergence_cleaning import clean_divergence


DivergenceStrategy = Literal["ct", "clean", "none"]


@dataclass(frozen=True)
class NumericalRecipe:
    """Runtime bundle for numerical scheme selection and stepping.
    
    DEPRECATED: Use Solver directly. Solver now has:
    - cfl_safety, dt_min, dt_max for timestep control
    - divergence_cleaning for constraint enforcement
    - step(state, model, geometry) that handles everything
    """

    solver: Solver
    time_controller: TimeController
    divergence_strategy: DivergenceStrategy = "none"
    use_checked_step: bool = False

    def __post_init__(self):
        warnings.warn(
            "NumericalRecipe is deprecated. Use Solver directly - it now has "
            "timestep control (cfl_safety, dt_min, dt_max) and divergence_cleaning. "
            "Call solver.step(state, model, geometry) instead.",
            DeprecationWarning,
            stacklevel=2
        )

    def validate(self, model, geometry) -> None:
        if self.divergence_strategy not in ("ct", "clean", "none"):
            raise ValueError(f"Unknown divergence strategy: {self.divergence_strategy}")

    def apply_constraints(self, state, model, geometry):
        state = model.apply_constraints(state, geometry)
        if self.divergence_strategy == "clean":
            state = state.replace(B=clean_divergence(state.B, geometry))
        return state

    def step(self, state, model, geometry):
        self.validate(model, geometry)
        dt = self.time_controller.compute_dt(state, model, geometry)
        if self.use_checked_step:
            next_state = self.solver.step_checked(state, dt, model, geometry)
        else:
            next_state = self.solver.advance(state, dt, model, geometry)
        return self.apply_constraints(next_state, model, geometry)
