# Resistive MHD (Lamy Ridge Model)

**File:** [resistive_mhd.py](../../resistive_mhd.py)

**Physics Level:** Macroscopic Fluid

## Key Features

- Flux function formulation with Grad-Shafranov evolution
- Chodura anomalous resistivity for rapid reconnection
- Circuit coupling with external coils
- 2D axisymmetric geometry

## Best For

- Engineering design
- Circuit optimization
- Formation dynamics

## Computational Cost

Low (Minutes)

## Equations

```
∂ψ/∂t + v·∇ψ = (η/μ₀)Δ*ψ + V_loop
V_bank = L_coil dI/dt + d/dt(∫M_plasma-coil dI_plasma)
```

## Usage

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

## JAX-FRC Framework Usage

```python
from jax_frc import Geometry, ResistiveMHD, RK4Solver, Simulation, TimeController
import jax.numpy as jnp

geometry = Geometry(
    coord_system='cylindrical',
    nr=64, nz=128,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

model = ResistiveMHD.from_config({
    'resistivity': {'type': 'chodura', 'eta_0': 1e-6, 'eta_anom': 1e-3}
})

solver = RK4Solver()
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final_state = sim.run_steps(500)
```

## Implementation Details

### Laplace Star Operator

The `laplace_star()` function computes the Δ* operator used in the Grad-Shafranov equation.

### Circuit Dynamics

The `circuit_dynamics()` function couples external coils to the plasma evolution.

### Chodura Resistivity

Anomalous resistivity model that mimics micro-turbulence effects at the plasma boundary. See [Physics Concepts](../reference/physics.md) for details.
