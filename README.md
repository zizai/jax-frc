# JAX Plasma Physics Simulation

A JAX-based implementation of three primary plasma physics models used in Field-Reversed Configuration (FRC) research: Resistive MHD (Lamy Ridge), Extended MHD (NIMROD), and Hybrid Kinetic-Fluid (HYM).

## Overview

This project provides GPU-accelerated implementations of plasma physics models using JAX's automatic differentiation and just-in-time compilation. The implementations are based on the theoretical framework described in `plasma_physics.md`.

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
