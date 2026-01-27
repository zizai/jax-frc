# Magnetic Diffusion Validation Notebooks Design

**Date**: 2026-01-27
**Status**: Implemented

## Overview

Two Jupyter notebooks for validating magnetic diffusion physics in the two limiting regimes of the magnetic Reynolds number.

## Notebooks

### 1. magnetic_diffusion.ipynb (Rm << 1)

**Physics**: Diffusion-dominated regime where magnetic field spreads through plasma like heat through a conductor.

**Key equation**: ∂B/∂t = η∇²B

**Analytic solution**: Spreading Gaussian
```
B_z(z,t) = B_peak × √(σ₀²/(σ₀² + 2ηt)) × exp(-z²/(2(σ₀² + 2ηt)))
```

**Validation metrics**:
- L2 error threshold: 5%
- Max relative error threshold: 10%

### 2. frozen_flux.ipynb (Rm >> 1)

**Physics**: Advection-dominated regime where magnetic field is frozen into the plasma (Alfvén's theorem).

**Key equation**: ∂B/∂t = ∇×(v×B)

**Analytic solution**: Flux conservation during radial expansion
```
B_φ(t) = B₀ × r₀/(r₀ + v_r×t)
```

**Validation metrics**:
- L2 error threshold: 5%
- Flux conservation threshold: 1%

## Structure (Both Notebooks)

1. YAML metadata header
2. Learning objectives (4-5 bullet points)
3. Physics background
   - Induction equation
   - Magnetic Reynolds number
   - Limiting regime physics
   - Analytic solution derivation
4. Configuration setup
   - Parameters
   - Geometry
   - Initial condition visualization
5. Simulation execution
   - Using lax.scan for efficiency
   - Progress snapshots
6. Analytic comparison
   - Overlay plots
   - Error distribution
   - Metrics summary
7. Time evolution animation
8. Interactive exploration
   - Time slider
   - Real-time metrics
9. Summary and next steps

## Dependencies

- ipywidgets
- matplotlib
- jax
- jax_frc (MagneticDiffusionConfiguration, FrozenFluxConfiguration)
- _shared.py utilities

## Files Created

- `notebooks/magnetic_diffusion.ipynb`
- `notebooks/frozen_flux.ipynb`
