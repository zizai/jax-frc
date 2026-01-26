# Physics Models

This project provides GPU-accelerated implementations of plasma physics models using JAX's automatic differentiation and just-in-time compilation.

## Available Models

| Model | Class | Solver | Use Case |
|-------|-------|--------|----------|
| Resistive MHD | `ResistiveMHD` | `RK4Solver` / `ImexSolver` | Circuit design, formation dynamics |
| Extended MHD | `ExtendedMHD` | `SemiImplicitSolver` | Hall physics, global stability |
| Hybrid Kinetic | `HybridKinetic` | `HybridSolver` | Kinetic stability, beam injection |
| Neutral Fluid | `NeutralFluid` | `RK4Solver` | Plasma-neutral coupling, FRC formation |
| Burning Plasma | `BurningPlasmaModel` | `RK4Solver` | Fusion burn, Q-factor, energy recovery |

## Model Comparison

| Feature | Resistive MHD | Extended MHD | Hybrid Kinetic | Neutral Fluid | Burning Plasma |
|---------|---------------|--------------|----------------|---------------|----------------|
| **Primary Equations** | E + v x B = eta J | E + v x B = eta J + (J x B)/(ne) | E <- Fluid; Ions <- Particles | Euler equations | MHD + Nuclear burn |
| **Numerical Approach** | Flux Function / Finite Volume | Finite Element + Semi-Implicit | Delta-f PIC | HLLE Riemann solver | Coupled multi-physics |
| **Computational Cost** | Low (Minutes) | High (Hours/Days) | Extreme (Days/Weeks) | Low (Minutes) | Medium (Hours) |
| **FRC Stability** | Fails (Predicts instability) | Good (Captures Hall effect) | Excellent (Captures FLR & Beams) | N/A (auxiliary) | Inherits from MHD core |
| **Best Use Case** | Circuit & Coil Design | Global Dynamics & Thermal Transport | Stability Limits & NBI Physics | Neutral fill, ionization | Fusion burn analysis |

## Detailed Documentation

- [Resistive MHD](resistive-mhd.md) - Single-fluid, flux function formulation
- [Extended MHD](extended-mhd.md) - Two-fluid with Hall effect
- [Hybrid Kinetic](hybrid-kinetic.md) - Kinetic ions + fluid electrons
- [Neutral Fluid](neutral-fluid.md) - Euler equations for neutrals + atomic rates
- [Burning Plasma](burning-plasma.md) - Multi-fuel burn with transport and energy recovery

## Limitations

1. **Resistive MHD:** Cannot predict FRC stability (fails on tilt mode)
2. **Extended MHD:** Misses betatron orbit resonance effects
3. **Hybrid Kinetic:** Computationally expensive for full formation cycles
4. **Neutral Fluid:** Single species (H only), no wall recycling
5. **Burning Plasma:** Alpha heating feedback to MHD not yet fully integrated
