# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based GPU-accelerated plasma physics simulation for Field-Reversed Configuration (FRC) research. Implements three physics models with increasing fidelity and computational cost.

## Commands

```bash
# Run all examples
python examples.py

# Run test suite
python test_simulations.py

# Run individual models
python -c "from resistive_mhd import run_simulation; run_simulation(steps=100)"
python -c "from extended_mhd import run_simulation; run_simulation(steps=100)"
python -c "from hybrid_kinetic import run_simulation; run_simulation(steps=100)"

# Run invariant tests
py -m pytest tests/ -v

# Run specific model tests
py -m pytest tests/test_resistive_mhd.py -v
py -m pytest tests/test_extended_mhd.py -v
py -m pytest tests/test_hybrid_kinetic.py -v

# Run with coverage (if pytest-cov installed)
py -m pytest tests/ --cov=. --cov-report=term-missing
```

## Architecture

### Three Physics Models (Increasing Fidelity)

1. **Resistive MHD** (`resistive_mhd.py`) - Single-fluid, flux function formulation
   - Solves for poloidal flux ψ(r,z) in 2D axisymmetric geometry
   - Key: `laplace_star()` computes Δ* operator, `circuit_dynamics()` couples external coils
   - Uses Chodura anomalous resistivity for rapid reconnection

2. **Extended MHD** (`extended_mhd.py`) - Two-fluid with Hall effect
   - Semi-implicit stepping for stiff Whistler waves
   - Key: `extended_ohm_law()` includes Hall term (J×B)/(ne)
   - Halo density model handles vacuum regions

3. **Hybrid Kinetic** (`hybrid_kinetic.py`) - Kinetic ions + fluid electrons
   - Delta-f PIC method reduces statistical noise
   - Key: `boris_push()` for particles, `weight_evolution()` for delta-f
   - Rigid rotor equilibrium distribution f₀

### Shared Utilities

`physics_utils.py` - Plasma parameter calculations (Alfvén speed, cyclotron frequency, beta, etc.) and numerical operators (gradient, divergence, curl, Laplacian)

### Simulation Pattern

All models use JAX's `lax.scan` for the main loop:
- State is an immutable tuple passed through `step()` function
- History accumulated via scan's output
- All compute functions are `@jax.jit` decorated

### Coordinate System

Cylindrical (r, θ, z) with 2D axisymmetric assumption. Array layout: `(spatial_r, spatial_z, [component])`.

### Physical Constants

```python
MU0 = 1.2566e-6   # Permeability of free space
QE = 1.602e-19    # Elementary charge
ME = 9.109e-31    # Electron mass
MI = 1.673e-27    # Ion mass (proton)
KB = 1.381e-23    # Boltzmann constant
```

### Typical Time Steps

- Resistive MHD: dt ~ 1e-4
- Extended MHD: dt ~ 1e-6 (Whistler constraint)
- Hybrid Kinetic: dt ~ 1e-8 (cyclotron constraint)


## Testing

Examples and validation tests are excluded from unittest.
