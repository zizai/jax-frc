"""Abstract base class for time integrators."""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel


class NumericalInstabilityError(Exception):
    """Raised when NaN or Inf values are detected in simulation state."""
    pass


class Solver(ABC):
    """Base class for time integration solvers - owns all numerics."""

    # Timestep control (absorbed from TimeController)
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3

    # Numerical options (absorbed from NumericalRecipe)
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"

    def _compute_dt(self, state: State, model: PhysicsModel, geometry) -> float:
        """Compute timestep from model CFL and config bounds."""
        dt_cfl = model.compute_stable_dt(state, geometry) * self.cfl_safety
        return float(jnp.clip(dt_cfl, self.dt_min, self.dt_max))

    def _apply_constraints(self, state: State, geometry) -> State:
        """Apply div(B)=0 cleaning based on divergence_cleaning setting."""
        if self.divergence_cleaning == "none":
            return state
        elif self.divergence_cleaning == "projection":
            from jax_frc.solvers.divergence_cleaning import clean_divergence
            B_clean = clean_divergence(state.B, geometry)
            return state.replace(B=B_clean)
        # Add more cleaning methods as needed
        return state

    @abstractmethod
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep without applying constraints."""
        raise NotImplementedError

    def step(self, state: State, model: PhysicsModel, geometry) -> State:
        """Complete step: compute dt, advance, apply constraints."""
        dt = self._compute_dt(state, model, geometry)
        new_state = self.advance(state, dt, model, geometry)
        new_state = self._apply_constraints(new_state, geometry)
        if self.use_checked_step:
            self._check_state(new_state)
        return new_state

    def step_with_dt(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep with explicit dt (legacy API)."""
        new_state = self.advance(state, dt, model, geometry)
        return model.apply_constraints(new_state, geometry)

    def step_checked(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep with NaN/Inf checking.

        Raises:
            NumericalInstabilityError: If NaN or Inf values are detected in the result.
        """
        new_state = self.advance(state, dt, model, geometry)
        self._check_state(new_state)
        return new_state

    def _check_state(self, state: State) -> None:
        """Check state for NaN/Inf values and raise error if found."""
        fields_to_check = [
            ("B", state.B),
            ("E", state.E),
            ("n", state.n),
            ("p", state.p),
            ("psi", state.psi),
            ("v", state.v),
        ]
        for name, field in fields_to_check:
            if field is not None:
                if jnp.any(jnp.isnan(field)):
                    raise NumericalInstabilityError(
                        f"NaN detected in {name} field at step {state.step}, t={state.time:.6e}"
                    )
                if jnp.any(jnp.isinf(field)):
                    raise NumericalInstabilityError(
                        f"Inf detected in {name} field at step {state.step}, t={state.time:.6e}"
                    )

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
        elif solver_type == "semi_implicit":
            from jax_frc.solvers.semi_implicit import SemiImplicitSolver
            return SemiImplicitSolver.from_config(config)
        elif solver_type == "hybrid":
            from jax_frc.solvers.semi_implicit import HybridSolver
            return HybridSolver()
        elif solver_type == "imex":
            from jax_frc.solvers.imex import ImexSolver, ImexConfig
            imex_config = ImexConfig(
                theta=float(config.get("theta", 1.0)),
                cg_tol=float(config.get("cg_tol", 1e-6)),
                cg_max_iter=int(config.get("cg_max_iter", 500)),
                cfl_factor=float(config.get("cfl_factor", 0.4))
            )
            return ImexSolver(config=imex_config)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
