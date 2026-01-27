# Getting Started

## Installation

```bash
pip install jax jaxlib numpy matplotlib
```

For GPU support:
```bash
pip install jax[cuda]
```

## Quick Start

### Programmatic Setup

```python
from jax_frc import Geometry
from jax_frc.core import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import EulerSolver
import jax.numpy as jnp

# Create 3D Cartesian geometry
geometry = Geometry(
    nx=32, ny=32, nz=64,
    x_min=-1.0, x_max=1.0,
    y_min=-1.0, y_max=1.0,
    z_min=-2.0, z_max=2.0,
    bc_x="periodic",
    bc_y="periodic",
    bc_z="dirichlet",
)

# Create initial state with a Gaussian magnetic field
x, y, z = geometry.x_grid, geometry.y_grid, geometry.z_grid
r_sq = x**2 + y**2 + z**2
B_z = jnp.exp(-r_sq / 0.1)

B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
B = B.at[:, :, :, 2].set(B_z)

state = State(
    B=B,
    E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
    n=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * 1e20,
    p=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * 1e4,
)

# Create physics model and solver
model = ResistiveMHD()
solver = EulerSolver()

# Time stepping
dt = 1e-6
for _ in range(100):
    state = solver.step(state, dt, model, geometry)

print(f"Final B_z max: {state.B[:, :, :, 2].max():.4f}")
```

### Configuration-Based Setup

Use predefined configurations for common validation cases:

```python
from jax_frc.configurations import MagneticDiffusionConfiguration
from jax_frc.solvers.explicit import EulerSolver

# Create configuration
config = MagneticDiffusionConfiguration(
    nx=64, ny=4, nz=64,  # Thin in y for 2D-like behavior
    extent=1.0,
    sigma=0.1,
    B_peak=1.0,
)

# Build components
geometry = config.build_geometry()
state = config.build_initial_state(geometry)
model = config.build_model()
solver = EulerSolver()

# Run simulation
dt = 0.3
for _ in range(100):
    state = solver.step(state, dt, model, geometry)
```

## Running Tests

```bash
# Run all tests
py -m pytest tests/ -v

# Run fast tests only
py -m pytest tests/ -k "not slow" -v

# Run specific 3D tests
py -m pytest tests/test_diffusion_3d.py -v
```

## Running Validation Cases

Validation cases are standalone Python scripts:

```bash
# Run magnetic diffusion validation
python validation/cases/analytic/magnetic_diffusion.py

# Each script produces an HTML report in validation/reports/
```

## Running Notebooks

Interactive Jupyter notebooks demonstrate physics concepts:

```bash
jupyter notebook notebooks/magnetic_diffusion.ipynb
```

## Key Concepts

### 3D Cartesian Coordinates

All simulations use 3D Cartesian coordinates (x, y, z):
- Scalars have shape `(nx, ny, nz)`
- Vectors have shape `(nx, ny, nz, 3)` with components `[Fx, Fy, Fz]`

For 2D-like simulations (e.g., diffusion in x-z plane), use a thin y dimension:
```python
geometry = Geometry(nx=64, ny=4, nz=64, bc_y="periodic")
```

### Boundary Conditions

Each axis can have different boundary conditions:
- `"periodic"` - Wrap-around boundaries
- `"dirichlet"` - Fixed value at boundary
- `"neumann"` - Zero gradient at boundary

### State Fields

The `State` class holds all field quantities:
- `B`: Magnetic field [T]
- `E`: Electric field [V/m]
- `n`: Number density [m^-3]
- `p`: Pressure [Pa]
- `v`: Velocity [m/s] (optional)
- `Te`, `Ti`: Electron/ion temperature [J] (optional)

## Physics Utilities

```python
from jax_frc.constants import MU0, PROTON_MASS
import jax.numpy as jnp

# Alfvén speed: v_A = B / sqrt(mu_0 * rho)
B = 1.0  # Tesla
n = 1e20  # m^-3
rho = n * PROTON_MASS
v_A = B / jnp.sqrt(MU0 * rho)
print(f"Alfvén speed: {v_A:.2e} m/s")
```
