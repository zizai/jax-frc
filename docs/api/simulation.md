# Core API

The simulation module provides the fundamental classes for building simulations.

## Simulation

Main orchestrator class using the builder pattern.

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

# Run simulation
final_state = sim.run(t_end=1.0)
```

### SimulationBuilder Methods

| Method | Description |
|--------|-------------|
| `geometry(g)` | Set computational geometry |
| `model(m)` | Set physics model |
| `solver(s)` | Set time integration solver |
| `initial_state(s)` | Set initial state |
| `phases(p)` | Set simulation phases (optional) |
| `callbacks(c)` | Set callbacks (optional) |
| `build()` | Create Simulation instance |

### Simulation Methods

| Method | Description |
|--------|-------------|
| `step()` | Advance one timestep |
| `run(t_end)` | Run until t_end |

### Presets

Pre-configured simulations for common test cases:

```python
from jax_frc.simulation.presets import create_magnetic_diffusion

sim = create_magnetic_diffusion(nx=64, ny=64, eta=1e-10)
sim.run(t_end=0.1)
```

## Geometry

Defines the 3D Cartesian computational domain.

```python
from jax_frc.simulation import Geometry

geometry = Geometry(
    nx=32, ny=32, nz=64,
    x_min=-1.0, x_max=1.0,
    y_min=-1.0, y_max=1.0,
    z_min=-2.0, z_max=2.0,
    bc_x="periodic",
    bc_y="periodic",
    bc_z="dirichlet",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nx`, `ny`, `nz` | int | Number of grid cells in each direction |
| `x_min`, `x_max` | float | Domain bounds in x (default: -1.0, 1.0) |
| `y_min`, `y_max` | float | Domain bounds in y (default: -1.0, 1.0) |
| `z_min`, `z_max` | float | Domain bounds in z (default: -1.0, 1.0) |
| `bc_x`, `bc_y`, `bc_z` | str | Boundary conditions: "periodic", "dirichlet", or "neumann" |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dx`, `dy`, `dz` | float | Grid spacing in each direction |
| `x`, `y`, `z` | Array | 1D arrays of cell-centered coordinates |
| `x_grid`, `y_grid`, `z_grid` | Array | 3D coordinate arrays, shape (nx, ny, nz) |
| `cell_volumes` | Array | Cell volumes (dx × dy × dz), shape (nx, ny, nz) |

### Array Layout

All fields use shape `(nx, ny, nz)` for scalars and `(nx, ny, nz, 3)` for vectors.

For 2D-like simulations, use a thin dimension (e.g., `ny=4`) with periodic boundary conditions in that direction.

## State

Container for simulation state variables.

```python
from jax_frc.simulation import State

# Create zero-initialized state
state = State.zeros(nx=32, ny=32, nz=64)

# Create with explicit fields
state = State(
    B=jnp.zeros((32, 32, 64, 3)),  # Magnetic field [T]
    E=jnp.zeros((32, 32, 64, 3)),  # Electric field [V/m]
    n=jnp.ones((32, 32, 64)),      # Number density [m^-3]
    p=jnp.ones((32, 32, 64)),      # Pressure [Pa]
)
```

### State Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `B` | (nx, ny, nz, 3) | Magnetic field [T] |
| `E` | (nx, ny, nz, 3) | Electric field [V/m] |
| `n` | (nx, ny, nz) | Number density [m^-3] |
| `p` | (nx, ny, nz) | Pressure [Pa] |
| `v` | (nx, ny, nz, 3) | Velocity [m/s] (optional) |
| `Te` | (nx, ny, nz) | Electron temperature [J] (optional) |
| `Ti` | (nx, ny, nz) | Ion temperature [J] (optional) |
| `particles` | ParticleState | Particle data for hybrid simulations (optional) |

### ParticleState

Holds particle data for hybrid kinetic simulations.

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | (n_particles, 3) | Particle positions |
| `v` | (n_particles, 3) | Particle velocities |
| `w` | (n_particles,) | Delta-f weights |
| `species` | str | Species identifier |

### Methods

| Method | Description |
|--------|-------------|
| `State.zeros(nx, ny, nz)` | Create zero-initialized state |
| `state.replace(**kwargs)` | Return new state with specified fields replaced |

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
| `step()` | Advance one timestep |
| `run_steps(n)` | Run n timesteps |
| `from_config(path)` | Load simulation from YAML config |

### Configuration-Based Setup

```python
sim = Simulation.from_config("configs/example_frc.yaml")
```
