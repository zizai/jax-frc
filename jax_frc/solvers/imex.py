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
        """Advance state by dt using IMEX Strang splitting.

        Strang splitting for 2nd-order accuracy:
        1. Half-step explicit (dt/2): advection, ideal terms
        2. Full implicit step (dt): resistive diffusion
        3. Half-step explicit (dt/2): advection, ideal terms

        Args:
            state: Current simulation state
            dt: Timestep
            model: Physics model
            geometry: Grid geometry

        Returns:
            Updated state at time t + dt
        """
        # Step 1: Half-step explicit
        state = self._explicit_half_step(state, dt / 2, model, geometry)

        # Step 2: Full implicit diffusion
        state = self._implicit_diffusion(state, dt, model, geometry)

        # Step 3: Half-step explicit
        state = self._explicit_half_step(state, dt / 2, model, geometry)

        # Update time and step count
        state = state.replace(
            time=state.time + dt,
            step=state.step + 1
        )

        return state

    def _explicit_half_step(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Advance explicit terms (advection, ideal MHD) by dt.

        This handles:
        - Advection: -v*grad(psi) for psi
        - Lorentz force: (JxB)/rho for velocity (if evolved)
        - Ideal induction: curl(v x B) for B (if B-based model)

        Note: Resistive diffusion is NOT included here (done implicitly).
        """
        # Apply constraints first
        state = model.apply_constraints(state, geometry)

        # For IMEX, compute only advective RHS (no diffusion)
        # Advection: -v*grad(psi)
        psi = state.psi
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        dr, dz = geometry.dr, geometry.dz

        # Compute gradient of psi (central differences)
        dpsi_dr = jnp.zeros_like(psi)
        dpsi_dr = dpsi_dr.at[1:-1, :].set(
            (psi[2:, :] - psi[:-2, :]) / (2 * dr)
        )
        dpsi_dz = jnp.zeros_like(psi)
        dpsi_dz = dpsi_dz.at[:, 1:-1].set(
            (psi[:, 2:] - psi[:, :-2]) / (2 * dz)
        )

        # Advection term: -v*grad(psi)
        advection = -(v_r * dpsi_dr + v_z * dpsi_dz)

        # Update psi with advection only (no diffusion)
        new_psi = state.psi + dt * advection

        new_state = state.replace(psi=new_psi)

        return model.apply_constraints(new_state, geometry)

    def _implicit_diffusion(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Solve implicit diffusion step for psi and B fields.

        For psi (resistive MHD):
            Solves: (I - theta*dt*(eta/mu0)*Delta*) psi^{n+1} = psi^n + ...

        For B components:
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

        # Solve implicit diffusion for psi (primary equation in resistive MHD)
        new_psi = self._solve_psi_diffusion(state.psi, dt, eta, geometry)

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

        return state.replace(psi=new_psi, B=B_new)

    def _solve_psi_diffusion(self, psi: Array, dt: float, eta: Array,
                             geometry: Geometry) -> Array:
        """Solve implicit diffusion for psi: (I - theta*dt*(eta/mu0)*nabla^2) psi = rhs.

        Uses Laplacian approximation for the implicit solve with Dirichlet BCs.
        """
        theta = self.config.theta
        dr, dz = geometry.dr, geometry.dz

        # Diffusion coefficient: D = eta/mu_0
        D = eta / MU0

        # Build operator for psi diffusion with boundary conditions
        # Boundaries stay fixed (identity), interior gets diffusion operator
        def psi_operator(psi_in: Array) -> Array:
            """Apply (I - theta*dt*D*nabla^2) to psi with Dirichlet BCs."""
            result = psi_in.copy()

            # Interior Laplacian
            lap_interior = (
                (psi_in[2:, 1:-1] - 2*psi_in[1:-1, 1:-1] + psi_in[:-2, 1:-1]) / dr**2 +
                (psi_in[1:-1, 2:] - 2*psi_in[1:-1, 1:-1] + psi_in[1:-1, :-2]) / dz**2
            )

            # Apply (I - theta*dt*D*nabla^2) only to interior
            D_interior = D[1:-1, 1:-1]
            result = result.at[1:-1, 1:-1].set(
                psi_in[1:-1, 1:-1] - theta * dt * D_interior * lap_interior
            )

            # Boundaries: identity (result[boundary] = psi_in[boundary])
            return result

        # Diagonal for preconditioner
        # Interior: 1 + theta*dt*D*(2/dr^2 + 2/dz^2), Boundary: 1
        diag = jnp.ones_like(psi)
        interior_diag = 1.0 + theta * dt * D[1:-1, 1:-1] * (2.0/dr**2 + 2.0/dz**2)
        diag = diag.at[1:-1, 1:-1].set(interior_diag)
        precond = jacobi_preconditioner(diag)

        # Build RHS
        if theta < 1.0:
            # Crank-Nicolson: RHS = psi^n + (1-theta)*dt*D*lap(psi^n)
            lap_psi = jnp.zeros_like(psi)
            lap_interior = (
                (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dr**2 +
                (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dz**2
            )
            lap_psi = lap_psi.at[1:-1, 1:-1].set(lap_interior)
            rhs = psi + (1 - theta) * dt * D * lap_psi
        else:
            # Backward Euler: RHS = psi^n
            rhs = psi

        # Solve with CG
        result = conjugate_gradient(
            psi_operator, rhs,
            x0=psi,
            preconditioner=precond,
            tol=self.config.cg_tol,
            max_iter=self.config.cg_max_iter
        )

        return result.x

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
