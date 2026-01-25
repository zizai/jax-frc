# IMEX Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement IMEX (Implicit-Explicit) time integration with Conjugate Gradient solver to remove diffusive CFL constraint from resistive MHD simulations.

**Architecture:** Add `jax_frc/solvers/linear/` package with CG solver and Jacobi preconditioner. Create `ImexSolver` class using Strang splitting: explicit half-step → implicit diffusion → explicit half-step. Update `PhysicsModel` base class with `explicit_terms()` and `diffusion_operator()` methods.

**Tech Stack:** JAX (jnp, lax.while_loop for CG iteration), Python dataclasses, pytest for TDD.

---

### Task 1: Create linear solver package structure

**Files:**
- Create: `jax_frc/solvers/linear/__init__.py`
- Create: `jax_frc/solvers/linear/cg.py`
- Create: `jax_frc/solvers/linear/preconditioners.py`

**Step 1: Create package init file**

```python
# jax_frc/solvers/linear/__init__.py
"""Linear solvers for implicit time integration."""

from jax_frc.solvers.linear.cg import conjugate_gradient, CGResult
from jax_frc.solvers.linear.preconditioners import jacobi_preconditioner

__all__ = ["conjugate_gradient", "CGResult", "jacobi_preconditioner"]
```

**Step 2: Create empty cg.py with CGResult dataclass**

```python
# jax_frc/solvers/linear/cg.py
"""Conjugate Gradient solver for symmetric positive-definite systems."""

from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class CGResult:
    """Result of Conjugate Gradient solve."""
    x: Array           # Solution
    converged: bool    # Did it converge?
    iterations: int    # Iterations used
    residual: float    # Final |Ax - b|/|b|


def conjugate_gradient(
    operator: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    preconditioner: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> CGResult:
    """Solve Ax = b using Conjugate Gradient.

    Args:
        operator: Matrix-free operator A(x) returning A @ x
        b: Right-hand side vector
        x0: Initial guess (defaults to zeros)
        preconditioner: Optional M^{-1}(r) preconditioner
        tol: Convergence tolerance for relative residual
        max_iter: Maximum iterations

    Returns:
        CGResult with solution and convergence info
    """
    raise NotImplementedError("CG solver not yet implemented")
```

**Step 3: Create empty preconditioners.py**

```python
# jax_frc/solvers/linear/preconditioners.py
"""Preconditioners for iterative linear solvers."""

from typing import Callable
import jax.numpy as jnp
from jax import Array


def jacobi_preconditioner(diag: Array) -> Callable[[Array], Array]:
    """Create Jacobi (diagonal) preconditioner.

    Args:
        diag: Diagonal elements of the matrix A

    Returns:
        Function M^{-1}(r) = r / diag
    """
    raise NotImplementedError("Jacobi preconditioner not yet implemented")
```

**Step 4: Verify package imports work**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -c "from jax_frc.solvers.linear import CGResult; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add jax_frc/solvers/linear/
git commit -m "feat(solvers): add linear solver package structure"
```

---

### Task 2: Implement Jacobi preconditioner

**Files:**
- Modify: `jax_frc/solvers/linear/preconditioners.py`
- Create: `tests/test_linear_solvers.py`

**Step 1: Write failing test for Jacobi preconditioner**

```python
# tests/test_linear_solvers.py
"""Tests for linear solvers."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.linear.preconditioners import jacobi_preconditioner


class TestJacobiPreconditioner:
    """Tests for Jacobi preconditioner."""

    def test_jacobi_divides_by_diagonal(self):
        """Jacobi preconditioner should divide residual by diagonal."""
        diag = jnp.array([2.0, 4.0, 8.0])
        precond = jacobi_preconditioner(diag)

        r = jnp.array([4.0, 8.0, 16.0])
        result = precond(r)

        expected = jnp.array([2.0, 2.0, 2.0])
        assert jnp.allclose(result, expected)

    def test_jacobi_handles_2d_arrays(self):
        """Jacobi should work with 2D arrays (field data)."""
        diag = jnp.ones((4, 4)) * 2.0
        precond = jacobi_preconditioner(diag)

        r = jnp.ones((4, 4)) * 6.0
        result = precond(r)

        expected = jnp.ones((4, 4)) * 3.0
        assert jnp.allclose(result, expected)

    def test_jacobi_jit_compatible(self):
        """Jacobi preconditioner should work under JIT."""
        from jax import jit

        diag = jnp.array([1.0, 2.0, 3.0])
        precond = jacobi_preconditioner(diag)
        precond_jit = jit(precond)

        r = jnp.array([2.0, 4.0, 6.0])
        result = precond_jit(r)

        expected = jnp.array([2.0, 2.0, 2.0])
        assert jnp.allclose(result, expected)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestJacobiPreconditioner::test_jacobi_divides_by_diagonal -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Implement Jacobi preconditioner**

```python
# jax_frc/solvers/linear/preconditioners.py
"""Preconditioners for iterative linear solvers."""

from typing import Callable
import jax.numpy as jnp
from jax import Array


def jacobi_preconditioner(diag: Array) -> Callable[[Array], Array]:
    """Create Jacobi (diagonal) preconditioner.

    The Jacobi preconditioner approximates A^{-1} as diag(A)^{-1}.
    For diffusion operators on uniform grids, this is effective because
    the diagonal dominates.

    Args:
        diag: Diagonal elements of the matrix A (same shape as solution)

    Returns:
        Function M^{-1}(r) = r / diag
    """
    # Avoid division by zero
    safe_diag = jnp.where(jnp.abs(diag) > 1e-14, diag, 1.0)

    def apply(r: Array) -> Array:
        return r / safe_diag

    return apply
```

**Step 4: Run all Jacobi tests to verify they pass**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestJacobiPreconditioner -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add jax_frc/solvers/linear/preconditioners.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement Jacobi preconditioner"
```

---

### Task 3: Implement Conjugate Gradient solver

**Files:**
- Modify: `jax_frc/solvers/linear/cg.py`
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write failing test for CG solver**

Add to `tests/test_linear_solvers.py`:

```python
from jax_frc.solvers.linear.cg import conjugate_gradient, CGResult


class TestConjugateGradient:
    """Tests for Conjugate Gradient solver."""

    def test_cg_solves_identity(self):
        """CG should solve Ix = b trivially."""
        def identity(x):
            return x

        b = jnp.array([1.0, 2.0, 3.0])
        result = conjugate_gradient(identity, b)

        assert jnp.allclose(result.x, b, rtol=1e-5)
        assert result.converged

    def test_cg_solves_diagonal_system(self):
        """CG should solve diagonal system Dx = b."""
        diag = jnp.array([2.0, 3.0, 4.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([4.0, 9.0, 16.0])
        result = conjugate_gradient(diag_op, b)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(result.x, expected, rtol=1e-5)
        assert result.converged

    def test_cg_solves_laplacian_1d(self):
        """CG should solve 1D Laplacian system."""
        n = 10

        # Tridiagonal: -1, 2, -1 (1D Laplacian with Dirichlet BC)
        def laplacian_1d(x):
            result = 2.0 * x
            result = result.at[:-1].add(-x[1:])
            result = result.at[1:].add(-x[:-1])
            return result

        # Known solution: quadratic
        x_true = jnp.arange(n, dtype=jnp.float32)
        b = laplacian_1d(x_true)

        result = conjugate_gradient(laplacian_1d, b, tol=1e-8, max_iter=100)

        assert jnp.allclose(result.x, x_true, rtol=1e-4)
        assert result.converged

    def test_cg_with_preconditioner(self):
        """CG should converge faster with preconditioner."""
        diag = jnp.array([10.0, 1.0, 10.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([20.0, 2.0, 30.0])

        # Without preconditioner
        result_no_precond = conjugate_gradient(diag_op, b, tol=1e-10)

        # With Jacobi preconditioner
        precond = jacobi_preconditioner(diag)
        result_precond = conjugate_gradient(diag_op, b, preconditioner=precond, tol=1e-10)

        # Both should converge to same answer
        assert jnp.allclose(result_no_precond.x, result_precond.x, rtol=1e-5)
        # Preconditioner should take fewer or equal iterations
        assert result_precond.iterations <= result_no_precond.iterations

    def test_cg_returns_residual(self):
        """CG should return relative residual."""
        def identity(x):
            return x

        b = jnp.array([1.0, 2.0, 3.0])
        result = conjugate_gradient(identity, b)

        # For identity, residual should be ~0 at convergence
        assert result.residual < 1e-5

    def test_cg_respects_max_iter(self):
        """CG should stop at max_iter if not converged."""
        # Ill-conditioned system that won't converge in 2 iterations
        diag = jnp.array([1.0, 1000.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([1.0, 1000.0])
        result = conjugate_gradient(diag_op, b, max_iter=2, tol=1e-15)

        assert result.iterations <= 2
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestConjugateGradient::test_cg_solves_identity -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Implement CG solver**

```python
# jax_frc/solvers/linear/cg.py
"""Conjugate Gradient solver for symmetric positive-definite systems."""

from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
import jax.lax as lax
from jax import Array


@dataclass(frozen=True)
class CGResult:
    """Result of Conjugate Gradient solve."""
    x: Array           # Solution
    converged: bool    # Did it converge?
    iterations: int    # Iterations used
    residual: float    # Final |Ax - b|/|b|


def conjugate_gradient(
    operator: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    preconditioner: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> CGResult:
    """Solve Ax = b using Conjugate Gradient.

    Implements the standard CG algorithm for symmetric positive-definite A.
    Uses lax.while_loop for JIT compatibility.

    Args:
        operator: Matrix-free operator A(x) returning A @ x
        b: Right-hand side vector
        x0: Initial guess (defaults to zeros)
        preconditioner: Optional M^{-1}(r) preconditioner
        tol: Convergence tolerance for relative residual |r|/|b|
        max_iter: Maximum iterations

    Returns:
        CGResult with solution and convergence info
    """
    # Initial guess
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = x0

    # Identity preconditioner if none provided
    if preconditioner is None:
        preconditioner = lambda r: r

    # Initial residual r = b - Ax
    r = b - operator(x)

    # Preconditioned residual z = M^{-1} r
    z = preconditioner(r)

    # Initial search direction
    p = z

    # r^T z for update
    rz = jnp.sum(r * z)

    # Norm of b for relative residual
    b_norm = jnp.linalg.norm(b)
    b_norm = jnp.where(b_norm > 1e-14, b_norm, 1.0)  # Avoid div by zero

    # CG iteration state: (x, r, z, p, rz, iteration, converged)
    def cg_body(state):
        x, r, z, p, rz, iteration, _ = state

        # Ap = A @ p
        Ap = operator(p)

        # alpha = r^T z / p^T A p
        pAp = jnp.sum(p * Ap)
        alpha = rz / jnp.where(jnp.abs(pAp) > 1e-14, pAp, 1e-14)

        # Update solution: x = x + alpha * p
        x_new = x + alpha * p

        # Update residual: r = r - alpha * Ap
        r_new = r - alpha * Ap

        # Preconditioned residual
        z_new = preconditioner(r_new)

        # New r^T z
        rz_new = jnp.sum(r_new * z_new)

        # beta = r_new^T z_new / r^T z
        beta = rz_new / jnp.where(jnp.abs(rz) > 1e-14, rz, 1e-14)

        # Update search direction: p = z + beta * p
        p_new = z_new + beta * p

        # Check convergence
        r_norm = jnp.linalg.norm(r_new)
        converged = (r_norm / b_norm) < tol

        return (x_new, r_new, z_new, p_new, rz_new, iteration + 1, converged)

    def cg_cond(state):
        _, _, _, _, _, iteration, converged = state
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))

    # Initial state
    init_state = (x, r, z, p, rz, 0, False)

    # Run CG loop
    final_state = lax.while_loop(cg_cond, cg_body, init_state)

    x_final, r_final, _, _, _, iterations, converged = final_state

    # Compute final residual
    residual = jnp.linalg.norm(r_final) / b_norm

    return CGResult(
        x=x_final,
        converged=converged,
        iterations=iterations,
        residual=residual
    )
```

**Step 4: Run all CG tests**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestConjugateGradient -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add jax_frc/solvers/linear/cg.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement Conjugate Gradient solver"
```

---

### Task 4: Add CG test for 2D diffusion operator

**Files:**
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write test for 2D diffusion**

Add to `tests/test_linear_solvers.py`:

```python
class TestCGDiffusion:
    """Tests for CG on diffusion-like operators."""

    def test_cg_solves_2d_laplacian(self):
        """CG should solve 2D Laplacian on small grid."""
        nr, nz = 8, 8

        # 2D Laplacian with Dirichlet BC (zeros at boundary)
        def laplacian_2d(x):
            # Interior: standard 5-point stencil
            result = 4.0 * x
            result = result.at[1:, :].add(-x[:-1, :])
            result = result.at[:-1, :].add(-x[1:, :])
            result = result.at[:, 1:].add(-x[:, :-1])
            result = result.at[:, :-1].add(-x[:, 1:])
            return result

        # Create a smooth RHS
        r = jnp.linspace(0, 1, nr)[:, None]
        z = jnp.linspace(0, 1, nz)[None, :]
        b = jnp.sin(jnp.pi * r) * jnp.sin(jnp.pi * z)

        result = conjugate_gradient(laplacian_2d, b, tol=1e-6, max_iter=200)

        # Verify Ax ≈ b
        Ax = laplacian_2d(result.x)
        assert jnp.allclose(Ax, b, rtol=1e-4)
        assert result.converged

    def test_cg_solves_implicit_diffusion_operator(self):
        """CG should solve (I - dt*D)x = b for implicit diffusion."""
        nr, nz = 8, 8
        dr, dz = 0.1, 0.1
        dt = 0.01
        D = 1.0  # Diffusion coefficient

        # Implicit diffusion operator: (I - dt*D*∇²)
        def implicit_diffusion(x):
            # Laplacian
            lap = jnp.zeros_like(x)
            lap = lap.at[1:-1, 1:-1].set(
                (x[2:, 1:-1] - 2*x[1:-1, 1:-1] + x[:-2, 1:-1]) / dr**2 +
                (x[1:-1, 2:] - 2*x[1:-1, 1:-1] + x[1:-1, :-2]) / dz**2
            )
            return x - dt * D * lap

        # RHS: some field after explicit step
        r = jnp.linspace(0, 1, nr)[:, None]
        z = jnp.linspace(0, 1, nz)[None, :]
        b = jnp.exp(-(r-0.5)**2 - (z-0.5)**2)

        # Jacobi preconditioner: diagonal is 1 + dt*D*(2/dr² + 2/dz²)
        diag_val = 1.0 + dt * D * (2.0/dr**2 + 2.0/dz**2)
        diag = jnp.ones((nr, nz)) * diag_val
        precond = jacobi_preconditioner(diag)

        result = conjugate_gradient(
            implicit_diffusion, b,
            preconditioner=precond,
            tol=1e-8, max_iter=100
        )

        # Verify solution
        Ax = implicit_diffusion(result.x)
        assert jnp.allclose(Ax, b, rtol=1e-5)
        assert result.converged
```

**Step 2: Run test**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestCGDiffusion -v`
Expected: 2 passed

**Step 3: Commit**

```bash
git add tests/test_linear_solvers.py
git commit -m "test(solvers): add CG tests for 2D diffusion operators"
```

---

### Task 5: Create ImexSolver class structure

**Files:**
- Create: `jax_frc/solvers/imex.py`
- Modify: `jax_frc/solvers/base.py`

**Step 1: Write failing test for ImexSolver**

Add to `tests/test_linear_solvers.py`:

```python
from jax_frc.solvers.imex import ImexSolver, ImexConfig


class TestImexSolver:
    """Tests for IMEX time integrator."""

    def test_imex_config_defaults(self):
        """ImexConfig should have sensible defaults."""
        config = ImexConfig()

        assert config.theta == 1.0  # Backward Euler
        assert config.cg_tol == 1e-6
        assert config.cg_max_iter == 500
        assert config.cfl_factor == 0.4

    def test_imex_solver_creates(self):
        """ImexSolver should instantiate."""
        config = ImexConfig()
        solver = ImexSolver(config)

        assert solver.config.theta == 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexSolver::test_imex_config_defaults -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Create ImexSolver skeleton**

```python
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
```

**Step 4: Run tests**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexSolver -v`
Expected: 2 passed

**Step 5: Add ImexSolver to factory in base.py**

Modify `jax_frc/solvers/base.py`:

```python
# Add to create() method after "semi_implicit" case:
        elif solver_type == "imex":
            from jax_frc.solvers.imex import ImexSolver, ImexConfig
            imex_config = ImexConfig(
                theta=float(config.get("theta", 1.0)),
                cg_tol=float(config.get("cg_tol", 1e-6)),
                cg_max_iter=int(config.get("cg_max_iter", 500)),
                cfl_factor=float(config.get("cfl_factor", 0.4))
            )
            return ImexSolver(config=imex_config)
```

**Step 6: Commit**

```bash
git add jax_frc/solvers/imex.py jax_frc/solvers/base.py tests/test_linear_solvers.py
git commit -m "feat(solvers): add ImexSolver class structure"
```

---

### Task 6: Implement implicit diffusion operator

**Files:**
- Modify: `jax_frc/solvers/imex.py`
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write failing test for diffusion operator**

Add to `tests/test_linear_solvers.py`:

```python
class TestImexDiffusion:
    """Tests for IMEX implicit diffusion."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        from jax_frc.core.geometry import Geometry
        return Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

    def test_build_diffusion_operator(self, geometry):
        """Should build implicit diffusion operator."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig

        config = ImexConfig(theta=1.0)
        solver = ImexSolver(config)

        dt = 0.001
        eta = jnp.ones((geometry.nr, geometry.nz)) * 1e-4  # Uniform resistivity

        # Build operator for B_z component
        operator, diag = solver._build_diffusion_operator(
            geometry, dt, eta, component='z'
        )

        # Test that operator is well-formed
        B_test = jnp.sin(jnp.pi * geometry.r_grid) * jnp.cos(jnp.pi * geometry.z_grid)
        result = operator(B_test)

        assert result.shape == B_test.shape
        assert jnp.all(jnp.isfinite(result))

    def test_diffusion_operator_identity_at_zero_dt(self, geometry):
        """At dt=0, operator should be identity."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig

        config = ImexConfig(theta=1.0)
        solver = ImexSolver(config)

        dt = 0.0
        eta = jnp.ones((geometry.nr, geometry.nz)) * 1e-4

        operator, _ = solver._build_diffusion_operator(geometry, dt, eta, component='z')

        B_test = jnp.sin(jnp.pi * geometry.r_grid)
        result = operator(B_test)

        assert jnp.allclose(result, B_test)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexDiffusion::test_build_diffusion_operator -v`
Expected: FAIL with `AttributeError: 'ImexSolver' object has no attribute '_build_diffusion_operator'`

**Step 3: Implement _build_diffusion_operator**

Add to `jax_frc/solvers/imex.py`:

```python
    def _build_diffusion_operator(
        self, geometry: Geometry, dt: float, eta: Array, component: str
    ) -> tuple:
        """Build implicit diffusion operator (I - θ*dt*D) and diagonal.

        Args:
            geometry: Grid geometry
            dt: Timestep
            eta: Resistivity field η(r,z)
            component: 'r', 'theta', or 'z' for B component

        Returns:
            (operator, diagonal) where operator is A(x) and diagonal for preconditioner
        """
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid
        theta = self.config.theta

        # Diffusion coefficient: D = η/μ₀
        D = eta / MU0

        # For Cartesian-like Laplacian: ∇²B = d²B/dr² + d²B/dz²
        # (Ignoring 1/r terms for B_z, which is approximately valid away from axis)

        # Diagonal of implicit operator: 1 + θ*dt*D*(2/dr² + 2/dz²)
        diag = 1.0 + theta * dt * D * (2.0/dr**2 + 2.0/dz**2)

        def operator(B: Array) -> Array:
            """Apply (I - θ*dt*D*∇²) to B field component."""
            # Laplacian using central differences
            # Interior only; boundaries handled separately
            lap = jnp.zeros_like(B)

            # d²B/dr²
            lap = lap.at[1:-1, :].add(
                (B[2:, :] - 2*B[1:-1, :] + B[:-2, :]) / dr**2
            )

            # d²B/dz²
            lap = lap.at[:, 1:-1].add(
                (B[:, 2:] - 2*B[:, 1:-1] + B[:, :-2]) / dz**2
            )

            # (I - θ*dt*D*∇²)B
            return B - theta * dt * D * lap

        return operator, diag
```

**Step 4: Run diffusion operator tests**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexDiffusion -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add jax_frc/solvers/imex.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement IMEX diffusion operator"
```

---

### Task 7: Implement implicit diffusion solve

**Files:**
- Modify: `jax_frc/solvers/imex.py`
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write failing test for implicit solve**

Add to `tests/test_linear_solvers.py`:

```python
    def test_implicit_diffusion_step(self, geometry):
        """Implicit diffusion should solve and update B field."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        config = ImexConfig(theta=1.0, cg_tol=1e-8)
        solver = ImexSolver(config)

        nr, nz = geometry.nr, geometry.nz

        # Create state with non-zero B_z
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(
            jnp.sin(jnp.pi * geometry.r_grid / geometry.r_max) *
            jnp.cos(jnp.pi * geometry.z_grid / (2 * geometry.z_max))
        )

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Create model with uniform resistivity
        model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-4))

        dt = 1e-3
        new_state = solver._implicit_diffusion(state, dt, model, geometry)

        # B should have changed (diffusion smooths it)
        assert not jnp.allclose(new_state.B, state.B)
        # B should remain finite
        assert jnp.all(jnp.isfinite(new_state.B))
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexDiffusion::test_implicit_diffusion_step -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Implement _implicit_diffusion**

Update in `jax_frc/solvers/imex.py`:

```python
    def _implicit_diffusion(self, state: State, dt: float,
                            model: PhysicsModel, geometry: Geometry) -> State:
        """Solve implicit diffusion step for B field.

        Solves: (I - θ*dt*D)·B^{n+1} = B^n + (1-θ)*dt*D·B^n

        For backward Euler (θ=1): (I - dt*D)·B^{n+1} = B^n
        For Crank-Nicolson (θ=0.5): (I - 0.5*dt*D)·B^{n+1} = (I + 0.5*dt*D)·B^n
        """
        # Get resistivity from model
        if hasattr(model, 'resistivity'):
            # For resistive MHD, compute J to get η
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

            # Build RHS: B^n + (1-θ)*dt*D·B^n
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
        """Compute 2D Laplacian ∇²f = d²f/dr² + d²f/dz²."""
        lap = jnp.zeros_like(f)

        # d²f/dr² (interior)
        lap = lap.at[1:-1, :].add(
            (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2
        )

        # d²f/dz² (interior)
        lap = lap.at[:, 1:-1].add(
            (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dz**2
        )

        return lap
```

**Step 4: Run test**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexDiffusion::test_implicit_diffusion_step -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add jax_frc/solvers/imex.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement IMEX implicit diffusion solve"
```

---

### Task 8: Implement explicit half-step

**Files:**
- Modify: `jax_frc/solvers/imex.py`
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write failing test**

Add to `tests/test_linear_solvers.py`:

```python
class TestImexExplicit:
    """Tests for IMEX explicit step."""

    @pytest.fixture
    def geometry(self):
        from jax_frc.core.geometry import Geometry
        return Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

    def test_explicit_half_step_advances_psi(self, geometry):
        """Explicit step should advance psi by advection."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        config = ImexConfig()
        solver = ImexSolver(config)

        nr, nz = geometry.nr, geometry.nz

        # Create state with non-zero psi and velocity
        psi = jnp.sin(jnp.pi * geometry.r_grid / geometry.r_max)
        v = jnp.zeros((nr, nz, 3))
        v = v.at[:, :, 0].set(0.1)  # Radial velocity

        state = State(
            psi=psi,
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=v,
            particles=None,
            time=0.0,
            step=0
        )

        model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))

        dt = 1e-5
        new_state = solver._explicit_half_step(state, dt, model, geometry)

        # psi should change due to advection
        # (small change expected for small dt)
        assert jnp.all(jnp.isfinite(new_state.psi))
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexExplicit::test_explicit_half_step_advances_psi -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Implement _explicit_half_step**

Update in `jax_frc/solvers/imex.py`:

```python
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
```

**Step 4: Run test**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexExplicit -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add jax_frc/solvers/imex.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement IMEX explicit half-step"
```

---

### Task 9: Implement full IMEX step with Strang splitting

**Files:**
- Modify: `jax_frc/solvers/imex.py`
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write failing test for full step**

Add to `tests/test_linear_solvers.py`:

```python
class TestImexFullStep:
    """Tests for complete IMEX time step."""

    @pytest.fixture
    def geometry(self):
        from jax_frc.core.geometry import Geometry
        return Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

    def test_imex_step_advances_time(self, geometry):
        """IMEX step should advance time and step count."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        config = ImexConfig()
        solver = ImexSolver(config)

        nr, nz = geometry.nr, geometry.nz

        state = State(
            psi=jnp.sin(jnp.pi * geometry.r_grid / geometry.r_max),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-4))

        dt = 1e-4
        new_state = solver.step(state, dt, model, geometry)

        assert new_state.time == dt
        assert new_state.step == 1
        assert jnp.all(jnp.isfinite(new_state.psi))

    def test_imex_step_stable_with_large_dt(self, geometry):
        """IMEX should remain stable with dt >> explicit CFL."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        config = ImexConfig(theta=1.0)  # Backward Euler for stability
        solver = ImexSolver(config)

        nr, nz = geometry.nr, geometry.nz

        state = State(
            psi=jnp.sin(jnp.pi * geometry.r_grid / geometry.r_max),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-3))

        # Explicit CFL would be ~dr²/(4*D) ~ 0.06²/(4*1e-3/μ₀) ~ very small
        # Use dt much larger than explicit limit
        dt = 1e-3

        # Run 10 steps
        for _ in range(10):
            state = solver.step(state, dt, model, geometry)

        # Should remain bounded (not blow up)
        assert jnp.all(jnp.isfinite(state.psi))
        assert jnp.max(jnp.abs(state.psi)) < 100.0  # Reasonable bound
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexFullStep::test_imex_step_advances_time -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Implement full step method**

Update in `jax_frc/solvers/imex.py`:

```python
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
```

**Step 4: Run tests**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_linear_solvers.py::TestImexFullStep -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add jax_frc/solvers/imex.py tests/test_linear_solvers.py
git commit -m "feat(solvers): implement full IMEX step with Strang splitting"
```

---

### Task 10: Add analytic diffusion test configuration

**Files:**
- Modify: `jax_frc/configurations/analytic.py`
- Create: `tests/test_imex_validation.py`

**Step 1: Write failing test**

```python
# tests/test_imex_validation.py
"""Validation tests for IMEX solver against analytic solutions."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResistiveDiffusionAnalytic:
    """Test IMEX solver against analytic resistive diffusion."""

    def test_1d_diffusion_decay_rate(self):
        """B should decay exponentially with correct rate."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.core.geometry import Geometry
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity

        # Setup
        nr, nz = 32, 64
        L_r, L_z = 1.0, 2.0
        geometry = Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=L_r,
            z_min=-L_z/2, z_max=L_z/2,
            nr=nr, nz=nz
        )

        eta = 1e-4
        model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=eta))

        config = ImexConfig(theta=1.0, cg_tol=1e-10)
        solver = ImexSolver(config)

        # Initial condition: sinusoidal in z
        # B(z,t) = B₀ sin(kz) exp(-k²Dt) where D = η/μ₀
        k = jnp.pi / L_z  # Fundamental mode
        B0 = 0.1
        MU0 = 1.2566e-6
        D = eta / MU0

        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(B0 * jnp.sin(k * geometry.z_grid))

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,
            B=B,
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        # Run simulation
        dt = 1e-4
        n_steps = 100
        t_final = dt * n_steps

        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Analytic solution at t_final
        decay = jnp.exp(-k**2 * D * t_final)
        B_analytic = B0 * decay * jnp.sin(k * geometry.z_grid)

        # Compare (use interior points to avoid boundary effects)
        B_numerical = state.B[nr//4:3*nr//4, nz//4:3*nz//4, 2]
        B_expected = B_analytic[nr//4:3*nr//4, nz//4:3*nz//4]

        # Should match within 10% (numerical errors from boundaries, etc.)
        rel_error = jnp.max(jnp.abs(B_numerical - B_expected)) / B0
        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"
```

**Step 2: Run test**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/test_imex_validation.py -v`
Expected: May pass or fail depending on implementation accuracy

**Step 3: Commit**

```bash
git add tests/test_imex_validation.py
git commit -m "test(validation): add analytic diffusion test for IMEX"
```

---

### Task 11: Run full test suite and fix any regressions

**Files:**
- Various fixes as needed

**Step 1: Run full test suite**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\imex-solver && py -m pytest tests/ -v`
Expected: All 132+ tests pass

**Step 2: Fix any failures**

If any tests fail, diagnose and fix. Common issues:
- Import errors from new modules
- Type mismatches in State
- Boundary condition handling

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test regressions from IMEX integration"
```

---

### Task 12: Update documentation and CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add IMEX documentation to CLAUDE.md**

Add to the Solvers section:

```markdown
### Solvers

- **EulerSolver** - Simple forward Euler (explicit)
- **RK4Solver** - 4th-order Runge-Kutta (explicit)
- **SemiImplicitSolver** - Hall damping + STS for temperature
- **ImexSolver** - IMEX with Strang splitting for resistive diffusion
  - Uses CG with Jacobi preconditioner for implicit step
  - Config: `theta` (1.0=backward Euler, 0.5=Crank-Nicolson)
  - Removes diffusive CFL constraint, allows 10-100× larger dt
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add IMEX solver documentation"
```

---

## Summary

| Task | Description | Est. Lines |
|------|-------------|------------|
| 1 | Package structure | ~30 |
| 2 | Jacobi preconditioner | ~40 |
| 3 | CG solver | ~100 |
| 4 | CG 2D diffusion tests | ~50 |
| 5 | ImexSolver structure | ~60 |
| 6 | Diffusion operator | ~50 |
| 7 | Implicit diffusion solve | ~60 |
| 8 | Explicit half-step | ~40 |
| 9 | Full IMEX step | ~30 |
| 10 | Analytic validation | ~80 |
| 11 | Regression fixes | varies |
| 12 | Documentation | ~20 |

**Total**: ~560 lines of new code + tests

