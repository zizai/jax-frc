# Hybrid Kinetic-Fluid (HYM Model)

**File:** [hybrid_kinetic.py](../../hybrid_kinetic.py)

**Physics Level:** Kinetic Ions + Fluid Electrons

## Key Features

- Delta-f Particle-in-Cell (PIC) method
- Rigid rotor equilibrium distribution
- Boris particle pusher
- Weight evolution for noise reduction

## Best For

- Stability limits
- Neutral beam injection (NBI) physics

## Computational Cost

Extreme (Days/Weeks)

## Equations

```
dv_i/dt = (q/m_i)(E + v_i×B)
E = (J_total×B - J_i,kinetic×B)/(ne) - ∇p_e/(ne) + ηJ
dw/dt = -(1-w)d ln f₀/dt
```

## Usage

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

## JAX-FRC Framework Usage

```python
from jax_frc import Geometry, HybridKinetic, HybridSolver, Simulation, TimeController
import jax.numpy as jnp

geometry = Geometry(
    coord_system='cylindrical',
    nr=32, nz=64,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

model = HybridKinetic.from_config({
    'n_particles': 10000,
    'equilibrium': 'rigid_rotor',
    'delta_f': True
})

solver = HybridSolver()
time_controller = TimeController(cfl_safety=0.1, dt_max=1e-8)

sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: jnp.exp(-r**2 - z**2))
final_state = sim.run_steps(100)
```

## Implementation Details

### Boris Particle Pusher

The `boris_push()` function advances particle positions and velocities using the second-order accurate Boris algorithm, which preserves phase-space volume.

### Delta-f Method

The `weight_evolution()` function evolves particle weights according to the delta-f method, reducing statistical noise by simulating only deviations from the equilibrium distribution.

### Rigid Rotor Equilibrium

Analytical equilibrium distribution for FRCs:

```
f₀ = n₀(m/(2πT))^(3/2) exp(-m/(2T)(v_r² + (v_θ-Ωr)² + v_z²))
```

See [Physics Concepts](../reference/physics.md) for more details.
