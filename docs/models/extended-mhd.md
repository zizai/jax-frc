# Extended MHD (NIMROD Model)

**File:** [extended_mhd.py](../../extended_mhd.py)

**Physics Level:** Two-Fluid Fluid Dynamics

## Key Features

- Extended Ohm's law with Hall term
- Semi-implicit time stepping for Whistler waves
- Halo density model for vacuum handling
- High-order finite elements

## Best For

- Global stability analysis
- Translation
- Transport scaling

## Computational Cost

High (Hours/Days)

## Equations

```
E = -v×B + ηJ + (J×B)/(ne) - ∇p_e/(ne)
(I - Δt²L_Hall)ΔB^{n+1} = Explicit terms
```

## Usage

```python
from extended_mhd import run_simulation

b_final, history = run_simulation(
    steps=100,
    nx=32,
    ny=32,
    dt=1e-6,
    eta=1e-4
)
```

## JAX-FRC Framework Usage

```python
from jax_frc import Geometry, ExtendedMHD, SemiImplicitSolver, Simulation, TimeController
import jax.numpy as jnp

geometry = Geometry(
    coord_system='cylindrical',
    nr=32, nz=64,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

model = ExtendedMHD.from_config({
    'resistivity': {'type': 'spitzer', 'eta_0': 1e-4},
    'hall': {'enabled': True}
})

solver = SemiImplicitSolver()
time_controller = TimeController(cfl_safety=0.1, dt_max=1e-6)

sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: jnp.exp(-r**2 - z**2))
final_state = sim.run_steps(100)
```

## Implementation Details

### Extended Ohm's Law

The `extended_ohm_law()` function includes the Hall term (J×B)/(ne) which captures two-fluid effects.

### Semi-Implicit Stepping

Numerical technique to handle stiff Whistler waves without requiring extremely small time steps. See [Solvers](../api/solvers.md) for details.

### Halo Density Model

Handles vacuum regions by maintaining a minimum "halo" density, preventing division-by-zero in the Hall term.
