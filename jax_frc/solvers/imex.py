# jax_frc/solvers/imex.py
"""IMEX (Implicit-Explicit) time integration solver."""

from dataclasses import dataclass, field
from typing import Optional
import jax.numpy as jnp
from jax import Array

from jax_frc.solvers.base import Solver
from jax_frc.solvers.linear import conjugate_gradient, jacobi_preconditioner, CGResult
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.base import PhysicsModel

MU0 = 1.2566e-6


@dataclass
class ImexConfig:
    """Configuration for IMEX solver."""
    theta: float = 1.0           # 1.0=backward Euler, 0.5=Crank-Nicolson
    cg_tol: float = 1e-6         # CG convergence tolerance
    cg_max_iter: int = 500       # CG max iterations
    cfl_factor: float = 0.4      # Explicit CFL safety factor


@dataclass
class ImexSolver(Solver):
    """IMEX time integrator with Strang splitting.

    Splits physics into:
    - Explicit: advection, Lorentz force, ideal induction
    - Implicit: resistive diffusion (solved with CG)

    Uses Strang splitting for 2nd-order accuracy:
    1. Half-step explicit (dt/2)
    2. Full implicit diffusion (dt)
    3. Half-step explicit (dt/2)
    """

    config: ImexConfig = field(default_factory=ImexConfig)

    def step(self, state: State, dt: float, model: PhysicsModel, geometry: Geometry) -> State:
        """Advance state by dt using IMEX splitting."""
        raise NotImplementedError("IMEX step not yet implemented")

    def _explicit_half_step(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Advance explicit terms by dt."""
        raise NotImplementedError("Explicit step not yet implemented")

    def _implicit_diffusion(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Solve implicit diffusion step."""
        raise NotImplementedError("Implicit diffusion not yet implemented")
