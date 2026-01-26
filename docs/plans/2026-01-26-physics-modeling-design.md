# Physics Modeling Design: EM Coils, Belova Comparison, Translation Validation

**Date:** 2026-01-26
**Status:** Approved

## Overview

This design covers three interconnected physics modeling goals:

1. **EM Coil Models** - Analytical field generators for solenoids, mirror coils, and theta-pinch arrays
2. **Belova Comparison** - Framework for comparing resistive MHD vs hybrid kinetic during FRC merging
3. **Translation Validation** - Benchmark cases for FRC translation/acceleration physics

## 1. EM Coil Models

**Location:** `jax_frc/fields/coils.py`

### Architecture

```
CoilField (base protocol)
  - B_field(r, z, t) -> (B_r, B_z)
  - A_phi(r, z, t) -> scalar (vector potential)
  - supports time-varying currents I(t)
      |
      +-- Solenoid(length, radius, n_turns, I)
      |     Uniform B_z inside, fringe fields outside
      |
      +-- MirrorCoil(z_position, radius, I)
      |     Single loop field (elliptic integrals)
      |     Pair creates mirror geometry
      |
      +-- ThetaPinchArray(coil_positions, radii, currents)
            Superposition of loops
            currents can be time-dependent array
```

### Design Choices

- All functions pure and JIT-compatible (no Python loops on traced values)
- Elliptic integrals via `jax.scipy.special` for loop fields
- Time-dependent currents passed as callable `I(t)` or interpolated arrays
- Returns fields on simulation grid for easy addition to equilibrium

## 2. Belova Comparison Framework

**Location:** `jax_frc/comparisons/belova_merging.py`

**Reference:** [Belova et al. 2025 - Hybrid Simulations of FRC Merging and Compression](https://arxiv.org/abs/2501.03425)

### Architecture

```
BelovaComparisonSuite
  - run_resistive_mhd(params) -> MergingResult
  - run_hybrid_kinetic(params) -> MergingResult
  - compare(mhd_result, hybrid_result) -> ComparisonReport
```

### Shared Parameters (from paper)

- Initial FRC separation, velocity, elongation
- Compression drive profile
- Grid resolution, particle count (hybrid)

### Diagnostics Tracked

| Diagnostic | Description |
|------------|-------------|
| `null_separation(t)` | Distance between magnetic nulls vs time |
| `reconnection_rate(t)` | dψ/dt at X-point |
| `merge_time` | When null separation -> 0 |
| `E_magnetic(t)` | Total magnetic energy |
| `E_kinetic(t)` | Bulk flow kinetic energy |
| `E_thermal(t)` | Thermal/internal energy |

### Output Format

- Time series saved to HDF5 or NumPy archives
- Comparison plots generated via matplotlib (optional)
- Summary metrics in structured dict for programmatic access

## 3. Translation Validation

**Location:** `jax_frc/validation/translation.py`

### Three Validation Tiers

**Tier 1: Analytic Benchmarks**
- Rigid FRC in uniform gradient -> known acceleration
- Mirror force: F = -μ·∇B (magnetic moment conservation)
- Compare simulation vs analytic trajectory

**Tier 2: Model Comparison**
- Same IC, same coil fields, three models:
  - One-fluid resistive MHD
  - Extended MHD (Hall + electron pressure)
  - Coupled resistive MHD (plasma + neutrals)
- Track: position(t), velocity(t), flux loss, heating

**Tier 3: Staged Acceleration (theta-pinch array)**
- Programmable coil timing sequence
- FRC accelerated through multiple stages
- Validates field superposition + timing control

### Example Cases

| Case | Coils | Model | Purpose |
|------|-------|-------|---------|
| `mirror_push_analytic` | 2 mirror coils | Resistive MHD | Validate against analytic |
| `translation_mhd_comparison` | Solenoid + mirrors | All three | Model intercomparison |
| `staged_acceleration` | Theta-pinch array | Resistive MHD | Multi-stage timing |

### Key Metrics

- Axial position of flux centroid vs time
- Axial velocity vs time
- Flux conservation (ψ_max vs initial)
- Energy budget (magnetic, kinetic, thermal)

## 4. Integration Architecture

### External Fields in Models

Add `external_field: Optional[CoilField]` to model configs. Models add `B_external` to total field before computing RHS.

### Diagnostics Registry

New diagnostics (`null_separation`, `reconnection_rate`, `energy_partition`) register with existing probe system.

### Comparison Runner

Thin wrapper that runs same config with different models, collects results, generates comparison output.

## 5. File Organization

```
jax_frc/
├── fields/
│   ├── __init__.py
│   └── coils.py           # NEW: Solenoid, MirrorCoil, ThetaPinchArray
├── comparisons/
│   ├── __init__.py
│   └── belova_merging.py  # NEW: comparison suite
├── validation/
│   ├── __init__.py
│   └── translation.py     # NEW: benchmark cases
└── diagnostics/
    ├── merging.py         # EXISTS: add reconnection_rate
    └── energy.py          # NEW: energy partition tracking

examples/
├── belova_case1_resistive.yaml
├── belova_case1_hybrid.yaml
├── belova_comparison.yaml
├── mirror_push_analytic.yaml
├── translation_mhd_comparison.yaml
└── staged_acceleration.yaml
```

## 6. Testing Strategy

| Layer | Test Type | What's Verified |
|-------|-----------|-----------------|
| Coils | Unit | Field values match analytical formulas at known points |
| Coils | Property | Symmetry (B_r=0 on axis), divergence-free (∇·B=0) |
| Diagnostics | Unit | Energy conservation in closed system |
| Comparison | Integration | Both models run, produce compatible output shapes |
| Validation | Benchmark | Simulation matches analytic trajectory within tolerance |

## 7. Implementation Order

```
Phase 1: Coil fields (no simulation dependency)
   Solenoid -> MirrorCoil -> ThetaPinchArray -> tests

Phase 2: Model integration
   Add external_field to ResistiveMHD -> ExtendedMHD -> tests

Phase 3: Diagnostics
   reconnection_rate -> energy_partition -> tests

Phase 4: Belova comparison
   Comparison runner -> example configs -> validation run

Phase 5: Translation validation
   Analytic benchmark -> model comparison -> staged accel -> tests
```

### Dependencies

- Phase 2 depends on Phase 1
- Phases 3-5 can partially parallelize after Phase 2
