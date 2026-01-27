# Model Comparison

Detailed comparison of the three physics models.

## Feature Comparison

| Feature | Resistive MHD | Extended MHD | Hybrid Kinetic |
|---------|---------------|--------------|----------------|
| **Primary Equations** | $\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J}$ | $\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{\mathbf{J} \times \mathbf{B}}{ne}$ | $\mathbf{E} \leftarrow$ Fluid; Ions $\leftarrow$ Particles |
| **Numerical Approach** | Flux Function / Finite Volume | Finite Element + Semi-Implicit | Delta-f PIC |
| **Computational Cost** | Low (Minutes) | High (Hours/Days) | Extreme (Days/Weeks) |
| **FRC Stability** | Fails (Predicts instability) | Good (Captures Hall effect) | Excellent (Captures FLR & Beams) |
| **Best Use Case** | Circuit & Coil Design | Global Dynamics & Thermal Transport | Stability Limits & NBI Physics |

## Physics Captured

| Effect | Resistive MHD | Extended MHD | Hybrid Kinetic |
|--------|---------------|--------------|----------------|
| Resistive diffusion | Yes | Yes | Yes |
| MHD waves | Yes | Yes | Yes |
| Hall effect | No | Yes | Yes |
| FLR stabilization | No | Partial | Yes |
| Kinetic resonances | No | No | Yes |
| Beam physics | No | No | Yes |

## Computational Requirements

| Model | Typical Grid | Typical $\Delta t$ | Memory | Runtime |
|-------|--------------|------------|--------|---------|
| Resistive MHD | 64×128 | $10^{-4}$ s | ~100 MB | Minutes |
| Extended MHD | 32×64 | $10^{-6}$ s | ~1 GB | Hours |
| Hybrid Kinetic | 32×64 + 10k particles | $10^{-8}$ s | ~10 GB | Days |

## Use Case Decision Tree

```
Need kinetic effects (beams, resonances)?
├── Yes → Hybrid Kinetic
└── No → Need Hall stabilization?
    ├── Yes → Extended MHD
    └── No → Need fast engineering scoping?
        ├── Yes → Resistive MHD
        └── No → Extended MHD (for accuracy)
```

## Typical Workflow

1. **Initial Design**: Use Resistive MHD for rapid circuit/coil optimization
2. **Stability Analysis**: Use Extended MHD for global dynamics
3. **Detailed Physics**: Use Hybrid Kinetic for stability limits and beam studies

## Limitations

### Resistive MHD
- Cannot predict FRC stability (fails on tilt mode)
- Missing Hall physics critical for FRC confinement
- Only suitable for macroscopic flow and circuit design

### Extended MHD
- Misses betatron orbit resonance effects
- May underestimate kinetic stabilization
- Computationally expensive for parameter scans

### Hybrid Kinetic
- Computationally expensive for full formation cycles
- Requires careful noise management
- Memory intensive for large particle counts
