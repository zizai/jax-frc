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
        """Advance explicit terms (advection, ideal MHD) by dt.

        This handles:
        - Advection: -v·∇ψ for psi
        - Lorentz force: (J×B)/ρ for velocity (if evolved)
        - Ideal induction: ∇×(v×B) for B (if B-based model)

        Note: Resistive diffusion is NOT included here (done implicitly).
        """
        # Apply constraints first
        state = model.apply_constraints(state, geometry)

        # Get RHS from model (includes all terms)
        rhs = model.compute_rhs(state, geometry)

        # For IMEX, we only want the non-diffusive parts
        # The model's RHS includes diffusion, so we need to subtract it
        # For now, just use forward Euler on the full RHS
        # (The implicit step will correct the diffusion part)

        # Update psi (advection-diffusion for resistive MHD)
        # In IMEX, we advance advection explicitly
        new_psi = state.psi + dt * rhs.psi

        # Update time (partial)
        new_state = state.replace(
            psi=new_psi,
        )

        return model.apply_constraints(new_state, geometry)

    def _implicit_diffusion(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Solve implicit diffusion step for B field.

        Solves: (I - theta*dt*D)·B^{n+1} = B^n + (1-theta)*dt*D·B^n

        For backward Euler (theta=1): (I - dt*D)·B^{n+1} = B^n
        For Crank-Nicolson (theta=0.5): (I - 0.5*dt*D)·B^{n+1} = (I + 0.5*dt*D)·B^n
        """
        # Get resistivity from model
        if hasattr(model, 'resistivity'):
            # For resistive MHD, compute J to get eta
            j_phi = model._compute_j_phi(state.psi, geometry)
            eta = model.resistivity.compute(j_phi)
        else:
            # Default uniform resistivity
            eta = jnp.ones((geometry.nr, geometry.nz)) * 1e-6

        theta = self.config.theta

        # Solve for each B component
        B_new = jnp.zeros_like(state.B)

        for i, comp in enumerate(['r', 'theta', 'z']):
            B_comp = state.B[:, :, i]

            # Build operator and preconditioner
            operator, diag = self._build_diffusion_operator(geometry, dt, eta, comp)
            precond = jacobi_preconditioner(diag)

            # Build RHS: B^n + (1-theta)*dt*D·B^n
            if theta < 1.0:
                # Compute explicit diffusion term
                D = eta / MU0
                lap = self._laplacian(B_comp, geometry.dr, geometry.dz)
                rhs = B_comp + (1 - theta) * dt * D * lap
            else:
                # Backward Euler: RHS is just B^n
                rhs = B_comp

            # Solve with CG
            result = conjugate_gradient(
                operator, rhs,
                x0=B_comp,  # Use current B as initial guess
                preconditioner=precond,
                tol=self.config.cg_tol,
                max_iter=self.config.cg_max_iter
            )

            B_new = B_new.at[:, :, i].set(result.x)

        return state.replace(B=B_new)

    def _laplacian(self, f: Array, dr: float, dz: float) -> Array:
        """Compute 2D Laplacian nabla^2 f = d^2f/dr^2 + d^2f/dz^2."""
        lap = jnp.zeros_like(f)

        # d^2f/dr^2 (interior)
        lap = lap.at[1:-1, :].add(
            (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2
        )

        # d^2f/dz^2 (interior)
        lap = lap.at[:, 1:-1].add(
            (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dz**2
        )

        return lap

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
