# Solvers

Time integration methods for advancing the simulation state.

## Available Solvers

| Solver | Class | Description |
|--------|-------|-------------|
| Euler | `EulerSolver` | First-order explicit |
| RK4 | `RK4Solver` | Fourth-order Runge-Kutta |
| Semi-Implicit | `SemiImplicitSolver` | For stiff systems (Whistler waves) |
| Hybrid | `HybridSolver` | Particle + field integration |

## Usage

```python
from jax_frc.solvers import RK4Solver, SemiImplicitSolver, HybridSolver

# For resistive MHD
solver = RK4Solver()

# For extended MHD
solver = SemiImplicitSolver()

# For hybrid kinetic
solver = HybridSolver()
```

## TimeController

Adaptive timestep control based on CFL conditions.

```python
from jax_frc.solvers import TimeController

time_controller = TimeController(
    cfl_safety=0.25,
    dt_max=1e-4
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfl_safety` | float | CFL safety factor (0-1) |
| `dt_max` | float | Maximum allowed timestep |
| `dt_min` | float | Minimum allowed timestep |

## Typical Time Steps

Different physics models require different timestep constraints:

| Model | Typical dt | Constraint |
|-------|------------|------------|
| Resistive MHD | ~1e-4 | Resistive diffusion |
| Extended MHD | ~1e-6 | Whistler wave |
| Hybrid Kinetic | ~1e-8 | Ion cyclotron |

## Semi-Implicit Method

The semi-implicit solver handles stiff Whistler waves in extended MHD:

```
(I - Δt²L_Hall)ΔB^{n+1} = Explicit terms
```

This allows larger timesteps than a fully explicit method would permit.
