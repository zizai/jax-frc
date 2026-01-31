# Solvers

Time integration methods for advancing the simulation state. Solvers own all numerical concerns including timestep control, divergence cleaning, and stability checks.

## Available Solvers

| Solver | Class | Description |
|--------|-------|-------------|
| Euler | `EulerSolver` | First-order explicit |
| RK4 | `RK4Solver` | Fourth-order Runge-Kutta |
| Semi-Implicit | `SemiImplicitSolver` | For stiff systems (Whistler waves) |
| IMEX | `ImexSolver` | Implicit-explicit for resistive diffusion |
| Hybrid | `HybridSolver` | Particle + field integration |

## Solver Base Class

All solvers inherit from `Solver` and include timestep control:

```python
from jax_frc.solvers.explicit import RK4Solver

solver = RK4Solver(
    cfl_safety=0.5,           # CFL safety factor (0-1)
    dt_min=1e-12,             # Minimum timestep
    dt_max=1e-3,              # Maximum timestep
    use_checked_step=True,    # Enable NaN/Inf checking
    divergence_cleaning="projection"  # div(B)=0 enforcement
)

# New API: step() computes dt internally
new_state = solver.step(state, model, geometry)
```

### Solver Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfl_safety` | float | 0.5 | CFL safety factor (0-1) |
| `dt_min` | float | 1e-12 | Minimum allowed timestep |
| `dt_max` | float | 1e-3 | Maximum allowed timestep |
| `use_checked_step` | bool | True | Check for NaN/Inf values |
| `divergence_cleaning` | str | "projection" | Constraint method ("projection", "none") |

### Solver Methods

| Method | Description |
|--------|-------------|
| `step(state, model, geometry)` | Complete timestep (compute dt, advance, apply constraints) |
| `advance(state, dt, model, geometry)` | Advance by explicit dt (no constraints) |
| `_compute_dt(state, model, geometry)` | Compute stable timestep |
| `_apply_constraints(state, geometry)` | Apply div(B)=0 cleaning |

## Usage with Simulation

```python
from jax_frc.simulation import Simulation, Geometry, State
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import RK4Solver

sim = Simulation.builder() \
    .geometry(Geometry(nx=64, ny=64, nz=1)) \
    .model(ExtendedMHD(eta=1e-4)) \
    .solver(RK4Solver(cfl_safety=0.25)) \
    .initial_state(State.zeros(64, 64, 1)) \
    .build()

sim.run(t_end=1.0)
```

## Deprecated: TimeController

> **Note:** `TimeController` is deprecated. Use `Solver` attributes directly.

```python
# OLD (deprecated)
from jax_frc.solvers import TimeController
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

# NEW
solver = RK4Solver(cfl_safety=0.25, dt_max=1e-4)
```

## Typical Time Steps

Different physics models require different timestep constraints:

| Model | Solver | Typical dt | Constraint |
|-------|--------|------------|------------|
| Resistive MHD | RK4 | ~1e-4 | Resistive diffusion |
| Resistive MHD | IMEX | ~1e-3 | Advection only (diffusion implicit) |
| Extended MHD | Semi-Implicit | ~1e-6 | Whistler wave |
| Hybrid Kinetic | Hybrid | ~1e-8 | Ion cyclotron |

## Semi-Implicit Method

The semi-implicit solver handles stiff Whistler waves in extended MHD:

```
(I - Δt²L_Hall)ΔB^{n+1} = Explicit terms
```

This allows larger timesteps than a fully explicit method would permit.

## IMEX Solver

The IMEX (Implicit-Explicit) solver uses Strang splitting to handle stiff resistive diffusion implicitly while treating advection explicitly.

### Configuration

```python
from jax_frc.solvers import ImexSolver, ImexConfig

config = ImexConfig(
    theta=1.0,          # 1.0=backward Euler, 0.5=Crank-Nicolson
    cg_tol=1e-6,        # CG convergence tolerance
    cg_max_iter=500,    # CG max iterations
    cfl_factor=0.4      # Explicit CFL safety factor
)

solver = ImexSolver(config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `theta` | float | 1.0 | Implicit weight (1.0=backward Euler, 0.5=Crank-Nicolson) |
| `cg_tol` | float | 1e-6 | Conjugate Gradient convergence tolerance |
| `cg_max_iter` | int | 500 | Maximum CG iterations |
| `cfl_factor` | float | 0.4 | CFL safety factor for explicit terms |

### Strang Splitting

For 2nd-order temporal accuracy, the solver uses Strang splitting:

1. **Half-step explicit** (dt/2): advection, ideal induction
2. **Full implicit step** (dt): resistive diffusion
3. **Half-step explicit** (dt/2): advection, ideal induction

### Implicit Diffusion

Solves the system:

```
(I - θ·dt·D)·ψ^{n+1} = ψ^n + (1-θ)·dt·D·ψ^n
```

where D = (η/μ₀)∇² is the diffusion operator.

For spatially-varying resistivity η(x) (e.g., Chodura anomalous), the operator becomes:

```
D·ψ = ∇·(η∇ψ)/μ₀
```

### Benefits

- **Removes diffusive CFL constraint**: dt can be 10-100× larger than explicit
- **Handles Chodura resistivity**: Spatially varying η supported
- **2nd-order accuracy**: Via Strang splitting

### Example

```python
from jax_frc.solvers import ImexSolver, ImexConfig
from jax_frc.models import ResistiveMHD

# Configure IMEX solver
config = ImexConfig(theta=1.0, cg_tol=1e-8)
solver = ImexSolver(config=config)

# Use with resistive MHD (large timesteps now stable)
model = ResistiveMHD.from_config({...})
new_state = solver.step(state, dt=1e-4, model=model, geometry=geometry)
```

## Linear Solvers

The IMEX solver uses iterative linear solvers for the implicit step.

### Conjugate Gradient

Matrix-free CG solver for symmetric positive-definite systems:

```python
from jax_frc.solvers.linear import conjugate_gradient, CGResult

result = conjugate_gradient(
    operator=A,           # Matrix-free A(x) function
    b=rhs,                # Right-hand side
    x0=initial_guess,     # Optional initial guess
    preconditioner=M_inv, # Optional M^{-1}(r)
    tol=1e-6,
    max_iter=1000
)

# Result fields
result.x          # Solution
result.converged  # bool
result.iterations # int
result.residual   # |Ax - b|/|b|
```

### Jacobi Preconditioner

Simple diagonal preconditioner:

```python
from jax_frc.solvers.linear import jacobi_preconditioner

# Create preconditioner from diagonal of A
diag = 1 + theta * dt * (eta/mu0) * (2/dr² + 2/dz²)
precond = jacobi_preconditioner(diag)

# Use with CG
result = conjugate_gradient(A, b, preconditioner=precond)
```
