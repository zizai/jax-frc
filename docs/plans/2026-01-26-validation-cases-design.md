# Validation Cases Design for Formation/Merging Phase

**Date:** 2026-01-26
**Target:** Cylindrical MHD and Hall reconnection validation
**Regime:** Formation/merging phase (translation, collision, reconnection)

---

## Overview

Three new validation cases to fill gaps in the validation ladder between analytic tests (Level 1) and FRC-specific merging (Level 4). All cases use cylindrical coordinates to validate the actual operators used in FRC simulations.

| Case | Purpose | Key Physics | Primary Metric |
|------|---------|-------------|----------------|
| `cylindrical_shock.yaml` | MHD shock capturing in z-direction | Compound waves, jump conditions | Shock speed ±5% of analytic |
| `cylindrical_vortex.yaml` | 2D MHD dynamics in (r,z) | Current sheet formation, energy cascade | Peak current density timing ±10% |
| `cylindrical_gem.yaml` | Hall reconnection | Fast reconnection, Hall quadrupole | Peak reconnection rate ±10% |

---

## File Structure

```
jax_frc/validation/cases/
├── analytic/
│   ├── diffusion_slab.yaml      (existing)
│   └── cylindrical_shock.yaml   (new)
├── mhd_regression/
│   └── cylindrical_vortex.yaml  (new)
├── hall_reconnection/
│   └── cylindrical_gem.yaml     (new)
└── frc_merging/
    └── belova_*.yaml            (existing)
```

---

## Case 1: Cylindrical Shock Tube

### Physical Setup
- Z-directed shock tube along the axis (r-independent initial conditions)
- Domain: r ∈ [0.01, 0.5] m, z ∈ [-1, 1] m
- Resolution: nr=16 (minimal, since r-uniform), nz=512

### Initial Conditions (Brio-Wu Adapted)
```
Left (z < 0):   ρ=1.0, p=1.0, Bz=0.75, Br=1.0, vz=0
Right (z > 0):  ρ=0.125, p=0.1, Bz=0.75, Br=-1.0, vz=0
```
- Bz constant (guide field), Br reverses across interface
- γ = 2.0 (matches standard Brio-Wu)

### Boundary Conditions
- z: Dirichlet (fixed left/right states)
- r: Symmetry at axis, Neumann at outer wall

### Acceptance Criteria (at t=0.1 τ_A)

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Fast shock position | z = 0.45 | ±5% |
| Slow shock position | z = 0.28 | ±5% |
| Contact discontinuity position | z = 0.15 | ±3% |
| Post-fast-shock density | ρ = 0.265 | ±3% |
| Max ∇·B / max(B) | < 1e-10 | absolute |

### What This Validates
- Shock-capturing numerics in cylindrical operators
- Compound wave resolution (fast/slow shocks, contact, rarefaction)
- Divergence cleaning effectiveness

---

## Case 2: Cylindrical Vortex

### Physical Setup
- Cylindrical annulus to avoid axis singularity complications
- Domain: r ∈ [0.2, 1.2] m, z ∈ [0, 2π] m (periodic in z)
- Resolution: nr=256, nz=256

### Initial Conditions (Orszag-Tang Adapted)
```
vr(r,z) = -v0 * sin(z)
vz(r,z) = v0 * sin(2π(r - r_min)/(r_max - r_min))
Br(r,z) = -B0 * sin(z)
Bz(r,z) = B0 * sin(4π(r - r_min)/(r_max - r_min))
ρ = ρ0 (uniform)
p = p0 (uniform, β ~ 10/3)
```
- v0 = 1.0, B0 = 1.0, ρ0 = 25/(36π), p0 = 5/(12π)
- Mach ~ 1, Alfvén Mach ~ 1

### Boundary Conditions
- z: Periodic
- r: Conducting wall (Br=0) at both r_min and r_max

### Acceptance Criteria (at t=0.5 τ_A)

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Time of peak current density | t = 0.48 τ_A | ±10% |
| Peak \|J\| / initial \|J\|_max | ~3.2 | ±15% |
| Total energy conservation | ΔE/E_0 | < 1% |
| Magnetic energy at t=0.5 | ~65% of initial | ±10% |

### What This Validates
- Nonlinear MHD dynamics in cylindrical geometry
- Current sheet formation and thinning
- Energy partition between kinetic/magnetic

---

## Case 3: Cylindrical GEM Reconnection

### Physical Setup
- Harris-like current sheet in (r,z) plane
- Domain: r ∈ [0.01, 2.0] m, z ∈ [-π, π] m
- Resolution: nr=256, nz=512 (finer in z across current sheet)

### Initial Conditions (GEM Harris Sheet)
```
Br(r,z) = B0 * tanh(z/λ)
Bz = 0
Bθ = 0 (initially, Hall quadrupole develops)
n(r,z) = n0 * sech²(z/λ) + n_b
T = T0 (uniform, from pressure balance)
```
- λ = 0.5 (current sheet half-width = 0.5 d_i)
- n_b = 0.2 n0 (background density)
- d_i = c/ω_pi (ion inertial length, sets Hall scale)

### Perturbation (to Seed Reconnection)
```
δψ = ψ1 * cos(2πr/Lr) * cos(z/λ)
```
- ψ1 = 0.1 B0 λ (10% perturbation)

### Boundary Conditions
- z: Periodic (or conducting at ±π)
- r: Symmetry at axis, conducting at r_max

### Model Requirements
- Extended MHD with Hall term enabled
- Hall parameter: d_i/λ ~ 1 (kinetic scale resolved)

### Acceptance Criteria

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Peak reconnection rate (dψ/dt) | ~0.1 B0 vA | ±10% |
| Time to peak rate | ~15 τ_A | ±15% |
| Reconnected flux at t=25 τ_A | ~0.8 ψ_max | ±10% |
| Hall quadrupole Bθ amplitude | Detectable (>0.05 B0) | qualitative |
| Current layer thickness at peak | ~d_i | ±20% |

### What This Validates
- Hall term implementation correctness
- Fast reconnection rate (should exceed Sweet-Parker)
- Quadrupole out-of-plane field (Hall signature)
- Current layer structure relevant to FRC merging

---

## Implementation Requirements

### New Configuration Classes

| Class | Base | Key Additions |
|-------|------|---------------|
| `CylindricalShockConfiguration` | `AbstractConfiguration` | 1D-in-z initial conditions, Brio-Wu states |
| `CylindricalVortexConfiguration` | `AbstractConfiguration` | Annulus geometry, periodic z BC, OT-like ICs |
| `CylindricalGEMConfiguration` | `AbstractConfiguration` | Harris sheet equilibrium, perturbation seeding |

### New Reference Data

```
jax_frc/validation/references/
├── cylindrical_shock_profile.npz    # Analytic Riemann solution at t=0.1
├── cylindrical_vortex_energy.npz    # Energy partition time series
└── cylindrical_gem_flux.npz         # Reconnected flux vs time (from high-res run)
```

### New Metrics (extend `metrics.py`)

```python
shock_position_error()       # Detects discontinuity location vs reference
reconnection_rate()          # dψ_reconnected/dt at X-point
hall_quadrupole_amplitude()  # Max |Bθ| in reconnection region
current_layer_thickness()    # FWHM of |J| across sheet
```

### Test Files

```
tests/
├── test_cylindrical_shock.py
├── test_cylindrical_vortex.py
└── test_cylindrical_gem.py
```

### Dependencies on Existing Code
- All cases use existing `Geometry`, `State`, `operators.py`
- Shock/Vortex use `ResistiveMHD` model
- GEM uses `ExtendedMHD` model with Hall enabled

---

## Validation Ladder Integration

| Level | Case | Status | Run Time | Pass Criteria |
|-------|------|--------|----------|---------------|
| 1a | `diffusion_slab.yaml` | Existing | ~seconds | L2 error < 1e-4 |
| 1b | `cylindrical_shock.yaml` | **New** | ~1 min | Shock positions ±5% |
| 2 | `cylindrical_vortex.yaml` | **New** | ~5 min | Energy conservation <1%, peak J timing ±10% |
| 3 | `cylindrical_gem.yaml` | **New** | ~10 min | Reconnection rate ±10%, quadrupole present |
| 4 | `belova_case*.yaml` | Existing | ~30 min | Merge timing, separatrix evolution |

### CI Integration

```yaml
# Fast tier (every PR): Levels 1a, 1b
# Nightly tier: Levels 1-3
# Weekly tier: Full ladder including Level 4
```

### Validation Report Output
Each case generates:
- Pass/fail status with metric values
- Comparison plots (reference vs computed)
- Convergence data if multiple resolutions run

---

## Key Considerations

From `validation_guide.md`:
- **Cylindrical benchmarks** validate the actual operators used for FRC simulations (unlike Cartesian benchmarks)
- **Hall physics** is critical for merging—MHD alone overpredicts reconnection rates and produces incorrect layer structure
- **GEM case** is the community standard for Hall/two-fluid reconnection validation
- **Quantitative tolerances** (±5-15%) provide clear pass/fail criteria while allowing for numerical differences
