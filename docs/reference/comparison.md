# Model Comparison

Detailed comparison of the three physics models.

## Feature Comparison

| Feature | Resistive MHD | Extended MHD | Hybrid Kinetic |
|---------|---------------|--------------|----------------|
| **Primary Equations** | E + v×B = ηJ | E + v×B = ηJ + (J×B)/(ne) | E ← Fluid; Ions ← Particles |
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

| Model | Typical Grid | Typical dt | Memory | Runtime |
|-------|--------------|------------|--------|---------|
| Resistive MHD | 64×128 | 1e-4 | ~100 MB | Minutes |
| Extended MHD | 32×64 | 1e-6 | ~1 GB | Hours |
| Hybrid Kinetic | 32×64 + 10k particles | 1e-8 | ~10 GB | Days |

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
