# Core API

The core module provides the fundamental classes for building simulations.

## Geometry

Defines the computational domain and coordinate system.

```python
from jax_frc import Geometry

geometry = Geometry(
    coord_system='cylindrical',
    nr=32, nz=64,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `coord_system` | str | Coordinate system ('cylindrical') |
| `nr` | int | Number of radial grid points |
| `nz` | int | Number of axial grid points |
| `r_min` | float | Minimum radial coordinate |
| `r_max` | float | Maximum radial coordinate |
| `z_min` | float | Minimum axial coordinate |
| `z_max` | float | Maximum axial coordinate |

### Coordinate System

Cylindrical (r, θ, z) with 2D axisymmetric assumption. Array layout: `(spatial_r, spatial_z, [component])`.

## State

Container for simulation state variables.

```python
from jax_frc.core import State, ParticleState
```

### State

Holds field quantities (flux function, magnetic field, density, etc.).

### ParticleState

Holds particle data for hybrid kinetic simulations (positions, velocities, weights).

## Simulation

Main orchestrator class that combines geometry, physics model, solver, and time controller.

```python
from jax_frc import Simulation

sim = Simulation(
    geometry=geometry,
    model=model,
    solver=solver,
    time_controller=time_controller
)
```

### Methods

| Method | Description |
|--------|-------------|
| `initialize(psi_init)` | Initialize state with flux function |
| `step()` | Advance one timestep |
| `run_steps(n)` | Run n timesteps |
| `from_config(path)` | Load simulation from YAML config |

### Configuration-Based Setup

```python
sim = Simulation.from_config("configs/example_frc.yaml")
```
