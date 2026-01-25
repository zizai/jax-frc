# IMEX Solver Design for JAX-FRC

**Date**: 2026-01-26
**Status**: Approved
**Goal**: Add Implicit-Explicit (IMEX) time integration to handle stiff resistive diffusion

## Overview

The current codebase uses explicit time stepping with subcycling for stiff terms, leading to:
- 50-substep cap for Whistler waves in extended MHD
- Explicit subcycling for resistive diffusion
- Timestep constraints dominated by diffusive CFL: dt < 0.25·dr²·μ₀/η

IMEX splitting treats advection/ideal MHD explicitly while solving resistive diffusion implicitly, removing the diffusive CFL constraint.

## Architecture

```
jax_frc/solvers/
├── base.py              (existing Solver ABC)
├── explicit.py          (existing forward Euler)
├── semi_implicit.py     (existing Hall subcycling)
├── imex.py              (NEW: IMEX time stepper)
└── linear/
    ├── __init__.py
    ├── cg.py            (NEW: Conjugate Gradient solver)
    └── preconditioners.py (NEW: Jacobi, future: ILU)
```

## Component 1: Conjugate Gradient Solver

**File**: `jax_frc/solvers/linear/cg.py`

```python
@dataclass
class CGResult:
    x: jnp.ndarray       # Solution
    converged: bool      # Did it converge?
    iterations: int      # Iterations used
    residual: float      # Final |Ax - b|/|b|

def conjugate_gradient(
    operator: Callable[[jnp.ndarray], jnp.ndarray],  # A(x), matrix-free
    b: jnp.ndarray,                                   # Right-hand side
    x0: jnp.ndarray | None = None,                   # Initial guess
    preconditioner: Callable | None = None,          # M^{-1}(r)
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> CGResult
```

**Design decisions**:
- Matrix-free: operator `A(x)` is a pure function, not a sparse matrix
- Uses `jax.lax.while_loop` for JIT-compatible iteration
- Returns fixed-size struct for `lax.scan` compatibility

## Component 2: Jacobi Preconditioner

**File**: `jax_frc/solvers/linear/preconditioners.py`

```python
def jacobi_preconditioner(diag: jnp.ndarray) -> Callable:
    """Returns M^{-1}(r) = r / diag(A)"""
    def apply(r):
        return r / diag
    return apply
```

For diffusion operator on uniform grid, diagonal is approximately:
```
diag ≈ 1 + θ·dt·(η/μ₀)·(2/dr² + 2/dz²)
```

## Component 3: IMEX Solver

**File**: `jax_frc/solvers/imex.py`

```python
@dataclass
class ImexConfig:
    theta: float = 1.0          # 1.0=backward Euler, 0.5=Crank-Nicolson
    cg_tol: float = 1e-6
    cg_max_iter: int = 500
    cfl_factor: float = 0.4     # Explicit CFL safety factor

class ImexSolver(Solver):
    def __init__(self, config: ImexConfig, geometry: Geometry):
        ...

    def step(self, state: State, dt: float, physics: PhysicsModel) -> State:
        # Strang splitting for 2nd-order accuracy
        state = self._explicit_half_step(state, dt/2, physics)
        state = self._implicit_diffusion(state, dt, physics)
        state = self._explicit_half_step(state, dt/2, physics)
        return state
```

### Strang Splitting

For 2nd-order temporal accuracy:

1. **Half-step explicit** (dt/2): advection, Lorentz force, ideal induction
2. **Full implicit step** (dt): resistive diffusion
3. **Half-step explicit** (dt/2): advection, Lorentz force, ideal induction

### Implicit Diffusion Step

Solves the system:
```
(I - θ·dt·D)·B^{n+1} = B^n + (1-θ)·dt·D·B^n
```

Where D·B = (η/μ₀)·∇²B for each magnetic field component.

For spatially-varying η(x) (Chodura anomalous resistivity), the operator becomes:
```
D·B = ∇·(η∇B)/μ₀
```

This remains symmetric positive-definite, so CG still applies.

## Component 4: Model Interface Updates

**File**: `jax_frc/models/base.py`

```python
class PhysicsModel(ABC):
    # Existing methods...

    @abstractmethod
    def explicit_terms(self, state: State, geometry: Geometry) -> State:
        """Returns dState/dt for explicit terms (advection, Lorentz, pressure)"""

    @abstractmethod
    def diffusion_operator(self, state: State, geometry: Geometry) -> State:
        """Returns D·state for implicit diffusion terms"""

    @abstractmethod
    def get_resistivity(self, state: State, geometry: Geometry) -> jnp.ndarray:
        """Returns η(x) field for diffusion coefficient"""
```

### Resistive MHD Changes

- Move induction split: ideal part → `explicit_terms`, resistive part → `diffusion_operator`
- Remove explicit subcycling for diffusion
- Keep Chodura resistivity model, expose via `get_resistivity`

### Extended MHD Changes

- Hall term stays in `explicit_terms` with subcycling (upgrade to implicit later)
- Resistive diffusion moves to `diffusion_operator`
- Electron pressure gradient stays explicit

## Component 5: Analytic Test Configuration

**File**: `jax_frc/configurations/analytic.py`

```python
class ResistiveDiffusionConfiguration(AbstractConfiguration):
    """1D magnetic diffusion: B(x,t) = B₀·exp(-k²ηt/μ₀)·sin(kx)"""

    def analytic_solution(self, t: float) -> jnp.ndarray:
        decay = jnp.exp(-self.k**2 * self.eta * t / MU0)
        return self.B0 * decay * jnp.sin(self.k * self.x)

    def error_metric(self, numerical: jnp.ndarray, t: float) -> float:
        return jnp.max(jnp.abs(numerical - self.analytic_solution(t)))
```

## Testing Strategy

### Unit Tests (`tests/test_imex_solver.py`)

1. **CG solver correctness**: Solve known linear system, verify against direct solve
2. **Preconditioner effect**: Compare iteration count with/without Jacobi
3. **Diffusion accuracy**: 1D heat equation with analytic solution, verify 2nd-order convergence
4. **Strang splitting order**: Confirm O(dt²) error by halving dt and checking 4× error reduction

### Integration Tests (`tests/test_imex_integration.py`)

1. **Resistive decay**: Initialize sinusoidal B perturbation, verify exponential decay rate matches η/(μ₀L²)
2. **Large timestep stability**: Run with dt 100× larger than explicit CFL, confirm no blowup
3. **Conservation**: Total magnetic energy should decay at rate 2∫ηJ²dV (resistive dissipation)

### Regression Tests

Compare against existing explicit solver on identical problem (should match to truncation error).

## Implementation Order

1. CG solver + Jacobi preconditioner
2. ImexSolver with Strang splitting
3. Update PhysicsModel interface
4. Refactor resistive MHD to use new interface
5. Refactor extended MHD
6. Add analytic test configuration
7. Integration tests

## Expected Outcomes

- Remove diffusive CFL constraint (dt can be 10-100× larger)
- Remove 50-substep cap for resistive terms
- Foundation for future implicit Hall term
- 2nd-order temporal accuracy via Strang splitting

## Future Extensions

- Implicit Hall term (requires GMRES for non-symmetric system)
- Crank-Nicolson (θ=0.5) for better accuracy on smooth problems
- Multigrid preconditioner for larger grids
- Adaptive timestepping based on CG convergence rate
