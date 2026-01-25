# Physics Models

This project provides GPU-accelerated implementations of three plasma physics models using JAX's automatic differentiation and just-in-time compilation.

## Available Models

| Model | Class | Solver | Use Case |
|-------|-------|--------|----------|
| Resistive MHD | `ResistiveMHD` | `RK4Solver` | Circuit design, formation dynamics |
| Extended MHD | `ExtendedMHD` | `SemiImplicitSolver` | Hall physics, global stability |
| Hybrid Kinetic | `HybridKinetic` | `HybridSolver` | Kinetic stability, beam injection |

## Model Comparison

| Feature | Resistive MHD | Extended MHD | Hybrid Kinetic |
|---------|---------------|--------------|----------------|
| **Primary Equations** | E + v×B = ηJ | E + v×B = ηJ + (J×B)/(ne) | E ← Fluid; Ions ← Particles |
| **Numerical Approach** | Flux Function / Finite Volume | Finite Element + Semi-Implicit | Delta-f PIC |
| **Computational Cost** | Low (Minutes) | High (Hours/Days) | Extreme (Days/Weeks) |
| **FRC Stability** | Fails (Predicts instability) | Good (Captures Hall effect) | Excellent (Captures FLR & Beams) |
| **Best Use Case** | Circuit & Coil Design | Global Dynamics & Thermal Transport | Stability Limits & NBI Physics |

## Detailed Documentation

- [Resistive MHD](resistive-mhd.md) - Single-fluid, flux function formulation
- [Extended MHD](extended-mhd.md) - Two-fluid with Hall effect
- [Hybrid Kinetic](hybrid-kinetic.md) - Kinetic ions + fluid electrons

## Limitations

1. **Resistive MHD:** Cannot predict FRC stability (fails on tilt mode)
2. **Extended MHD:** Misses betatron orbit resonance effects
3. **Hybrid Kinetic:** Computationally expensive for full formation cycles
