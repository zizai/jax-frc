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
from jax_frc import Simulation, Geometry, ResistiveMHD, RK4Solver, TimeController
import jax.numpy as jnp

# Create geometry
geometry = Geometry(
    coord_system='cylindrical',
    nr=32, nz=64,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

# Create physics model
model = ResistiveMHD.from_config({
    'resistivity': {'type': 'chodura', 'eta_0': 1e-6, 'eta_anom': 1e-3}
})

# Create solver and time controller
solver = RK4Solver()
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

# Create and run simulation
sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))

# Run for 100 steps
final_state = sim.run_steps(100)
```

### Configuration-Based Setup

```python
from jax_frc import Simulation

# Load from YAML configuration
sim = Simulation.from_config("configs/example_frc.yaml")
sim.initialize(psi_init=lambda r, z: jnp.exp(-r**2 - z**2))
final_state = sim.run_steps(100)
```

## Running Examples

Run all examples:
```bash
python examples.py
```

## Individual Model Usage

### Resistive MHD
```python
from resistive_mhd import run_simulation

final_psi, final_I_coil, history = run_simulation(
    steps=500,
    nr=64,
    nz=128,
    V_bank=1000.0,
    L_coil=1e-6,
    M_plasma_coil=1e-7
)
```

### Extended MHD
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

### Hybrid Kinetic
```python
from hybrid_kinetic import run_simulation

x_final, v_final, w_final, history = run_simulation(
    steps=100,
    n_particles=10000,
    nr=32,
    nz=64,
    dt=1e-8,
    eta=1e-4
)
```

## Physics Utilities

```python
from physics_utils import (
    compute_alfven_speed,
    compute_cyclotron_frequency,
    compute_larmor_radius,
    compute_skin_depth,
    compute_beta
)

# Calculate plasma parameters
v_A = compute_alfven_speed(B=1.0, n=1e19)
omega_c = compute_cyclotron_frequency(B=1.0)
r_L = compute_larmor_radius(v_perp=1e5, B=1.0)
d_i = compute_skin_depth(n=1e19)
beta = compute_beta(n=1e19, T=100.0, B=1.0)
```

## Migration from Old Scripts

Use the migration script to run old-style simulations with the new framework:

```bash
python scripts/migrate_old_sims.py --model resistive_mhd --steps 100
python scripts/migrate_old_sims.py --model extended_mhd --steps 50
python scripts/migrate_old_sims.py --model hybrid_kinetic --steps 100 --particles 1000
```
