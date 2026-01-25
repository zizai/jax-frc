# JAX Plasma Physics Simulation

A JAX-based implementation of three primary plasma physics models used in Field-Reversed Configuration (FRC) research: Resistive MHD (Lamy Ridge), Extended MHD (NIMROD), and Hybrid Kinetic-Fluid (HYM).

## Overview

This project provides GPU-accelerated implementations of plasma physics models using JAX's automatic differentiation and just-in-time compilation. The implementations are based on the theoretical framework described in `plasma_physics.md`.

## JAX-FRC Framework (New)

The `jax_frc` package provides a modular OOP framework for building and running FRC simulations with swappable physics models, solvers, and diagnostics.

### Quick Start

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

### Package Structure

```
jax_frc/
├── core/
│   ├── geometry.py      # Computational domain and coordinates
│   ├── state.py         # State containers (State, ParticleState)
│   └── simulation.py    # Main orchestrator class
├── models/
│   ├── base.py          # PhysicsModel abstract base class
│   ├── resistive_mhd.py # Single-fluid resistive MHD
│   ├── extended_mhd.py  # Two-fluid Extended MHD with Hall term
│   ├── hybrid_kinetic.py# Hybrid kinetic with delta-f PIC
│   └── resistivity.py   # Spitzer and Chodura resistivity models
├── solvers/
│   ├── base.py          # Solver abstract base class
│   ├── explicit.py      # Euler and RK4 solvers
│   ├── semi_implicit.py # Semi-implicit and hybrid solvers
│   └── time_controller.py# Adaptive timestep control
├── boundaries/
│   ├── base.py          # BoundaryCondition base class
│   ├── conducting.py    # Conducting wall boundary
│   └── symmetry.py      # Symmetry axis boundary
├── equilibrium/
│   ├── grad_shafranov.py# Grad-Shafranov solver
│   └── rigid_rotor.py   # Rigid rotor equilibrium
├── diagnostics/
│   ├── probes.py        # Diagnostic probes (Flux, Energy, Beta, Current)
│   └── output.py        # HDF5 checkpoints and time history I/O
└── config/
    └── loader.py        # YAML configuration loading
```

### Available Physics Models

| Model | Class | Solver | Use Case |
|-------|-------|--------|----------|
| Resistive MHD | `ResistiveMHD` | `RK4Solver` | Circuit design, formation dynamics |
| Extended MHD | `ExtendedMHD` | `SemiImplicitSolver` | Hall physics, global stability |
| Hybrid Kinetic | `HybridKinetic` | `HybridSolver` | Kinetic stability, beam injection |

### Diagnostics

```python
from jax_frc.diagnostics import DiagnosticSet, save_time_history, save_checkpoint

# Create default diagnostic set
diagnostics = DiagnosticSet.default_set()

# Measure during simulation
for i in range(steps):
    sim.step()
    results = diagnostics.measure_all(sim.state, geometry)

# Save time history and checkpoint
save_time_history(diagnostics.get_history(), "output/history.csv")
save_checkpoint(sim.state, geometry, "output/checkpoint.h5")
```

### Migration from Old Scripts

Use the migration script to run old-style simulations with the new framework:

```bash
python scripts/migrate_old_sims.py --model resistive_mhd --steps 100
python scripts/migrate_old_sims.py --model extended_mhd --steps 50
python scripts/migrate_old_sims.py --model hybrid_kinetic --steps 100 --particles 1000
```

## Models

### 1. Resistive MHD (Lamy Ridge Model)

**File:** [resistive_mhd.py](resistive_mhd.py)

**Physics Level:** Macroscopic Fluid

**Key Features:**
- Flux function formulation with Grad-Shafranov evolution
- Chodura anomalous resistivity for rapid reconnection
- Circuit coupling with external coils
- 2D axisymmetric geometry

**Best For:** Engineering design, circuit optimization, formation dynamics

**Computational Cost:** Low (Minutes)

**Equations:**
```
∂ψ/∂t + v·∇ψ = (η/μ₀)Δ*ψ + V_loop
V_bank = L_coil dI/dt + d/dt(∫M_plasma-coil dI_plasma)
```

### 2. Extended MHD (NIMROD Model)

**File:** [extended_mhd.py](extended_mhd.py)

**Physics Level:** Two-Fluid Fluid Dynamics

**Key Features:**
- Extended Ohm's law with Hall term
- Semi-implicit time stepping for Whistler waves
- Halo density model for vacuum handling
- High-order finite elements

**Best For:** Global stability analysis, translation, transport scaling

**Computational Cost:** High (Hours/Days)

**Equations:**
```
E = -v×B + ηJ + (J×B)/(ne) - ∇p_e/(ne)
(I - Δt²L_Hall)ΔB^{n+1} = Explicit terms
```

### 3. Hybrid Kinetic-Fluid (HYM Model)

**File:** [hybrid_kinetic.py](hybrid_kinetic.py)

**Physics Level:** Kinetic Ions + Fluid Electrons

**Key Features:**
- Delta-f Particle-in-Cell (PIC) method
- Rigid rotor equilibrium distribution
- Boris particle pusher
- Weight evolution for noise reduction

**Best For:** Stability limits and neutral beam injection (NBI) physics

**Computational Cost:** Extreme (Days/Weeks)

**Equations:**
```
dv_i/dt = (q/m_i)(E + v_i×B)
E = (J_total×B - J_i,kinetic×B)/(ne) - ∇p_e/(ne) + ηJ
dw/dt = -(1-w)d ln f₀/dt
```

## Installation

```bash
pip install jax jaxlib numpy matplotlib
```

For GPU support:
```bash
pip install jax[cuda]
```

## Usage

### Running Examples

Run all examples:
```bash
python examples.py
```

### Individual Model Usage

#### Resistive MHD
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

#### Extended MHD
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

#### Hybrid Kinetic
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

### Physics Utilities

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

## Model Comparison

| Feature | Resistive MHD | Extended MHD | Hybrid Kinetic |
|---------|---------------|--------------|----------------|
| **Primary Equations** | E + v×B = ηJ | E + v×B = ηJ + (J×B)/(ne) | E ← Fluid; Ions ← Particles |
| **Numerical Approach** | Flux Function / Finite Volume | Finite Element + Semi-Implicit | Delta-f PIC |
| **Computational Cost** | Low (Minutes) | High (Hours/Days) | Extreme (Days/Weeks) |
| **FRC Stability** | Fails (Predicts instability) | Good (Captures Hall effect) | Excellent (Captures FLR & Beams) |
| **Best Use Case** | Circuit & Coil Design | Global Dynamics & Thermal Transport | Stability Limits & NBI Physics |

## Key Physics Concepts

### Flux Function Formulation
In resistive MHD, we solve for the poloidal flux ψ(r,z) instead of the full magnetic field vector. This reduces the problem to 2D axisymmetric geometry.

### Chodura Resistivity
Anomalous resistivity model for FRC formation that mimics micro-turbulence effects at the plasma boundary.

### Hall Term
Two-fluid effect that separates electron and ion motion, crucial for capturing kinetic stabilization in FRCs.

### Semi-Implicit Stepping
Numerical technique to handle stiff Whistler waves in extended MHD without requiring extremely small time steps.

### Delta-f PIC Method
Particle-in-Cell method that simulates deviations from an equilibrium distribution, reducing noise by a factor of 1/δf.

### Rigid Rotor Equilibrium
Analytical equilibrium distribution for FRCs: f₀ = n₀(m/(2πT))^(3/2)exp(-m/(2T)(v_r² + (v_θ-Ωr)² + v_z²))

## File Structure

```
jax-fusion/
├── resistive_mhd.py       # Resistive MHD implementation
├── extended_mhd.py         # Extended MHD implementation
├── hybrid_kinetic.py       # Hybrid Kinetic-Fluid implementation
├── physics_utils.py        # Physics utility functions
├── examples.py             # Example usage scripts
├── plasma_physics.md       # Theoretical framework
└── README.md               # This file
```

## Performance Considerations

- **JAX JIT Compilation:** All functions are JIT-compiled for optimal performance
- **GPU Acceleration:** Automatically uses GPU if available
- **Vectorization:** Operations are vectorized using JAX's array operations
- **Memory:** Hybrid kinetic model requires significant memory for particle data

## Limitations

1. **Resistive MHD:** Cannot predict FRC stability (fails on tilt mode)
2. **Extended MHD:** Misses betatron orbit resonance effects
3. **Hybrid Kinetic:** Computationally expensive for full formation cycles

## References

Theoretical framework based on FRC research and the following codes:
- **Lamy Ridge:** Resistive MHD code for FRC formation
- **NIMROD:** Extended MHD code for global stability
- **HYM:** Hybrid kinetic code for stability and beam physics

## Contributing

Contributions are welcome! Areas for improvement:
- 3D geometry support
- More sophisticated boundary conditions
- Advanced particle weighting schemes
- Visualization tools
- Benchmarking against experimental data

## License

This project is provided for educational and research purposes.

## Contact

For questions or issues, please open an issue on the project repository.
