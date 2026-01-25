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

    def _build_diffusion_operator(
        self, geometry: Geometry, dt: float, eta: Array, component: str
    ) -> tuple:
        """Build implicit diffusion operator (I - theta*dt*D) and diagonal.

        Args:
            geometry: Grid geometry
            dt: Timestep
            eta: Resistivity field eta(r,z)
            component: 'r', 'theta', or 'z' for B component

        Returns:
            (operator, diagonal) where operator is A(x) and diagonal for preconditioner
        """
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid
        theta = self.config.theta

        # Diffusion coefficient: D = eta/mu_0
        D = eta / MU0

        # For Cartesian-like Laplacian: nabla^2 B = d^2B/dr^2 + d^2B/dz^2
        # (Ignoring 1/r terms for B_z, which is approximately valid away from axis)

        # Diagonal of implicit operator: 1 + theta*dt*D*(2/dr^2 + 2/dz^2)
        diag = 1.0 + theta * dt * D * (2.0/dr**2 + 2.0/dz**2)

        def operator(B: Array) -> Array:
            """Apply (I - theta*dt*D*nabla^2) to B field component."""
            # Laplacian using central differences
            # Interior only; boundaries handled separately
            lap = jnp.zeros_like(B)

            # d^2B/dr^2
            lap = lap.at[1:-1, :].add(
                (B[2:, :] - 2*B[1:-1, :] + B[:-2, :]) / dr**2
            )

            # d^2B/dz^2
            lap = lap.at[:, 1:-1].add(
                (B[:, 2:] - 2*B[:, 1:-1] + B[:, :-2]) / dz**2
            )

            # (I - theta*dt*D*nabla^2)B
            return B - theta * dt * D * lap

        return operator, diag
