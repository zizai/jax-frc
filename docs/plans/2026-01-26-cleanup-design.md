# Codebase Cleanup Design

## Overview

Remove legacy root-level implementation files and consolidate test structure into a single `tests/` directory.

## Goals

1. **Clean up legacy code** - Remove old root-level standalone implementations now that proper class-based versions exist in `jax_frc/models/`
2. **Consolidate test structure** - Organize all tests into `tests/` directory

## Changes

### 1. Delete Root-Level Legacy Files

| File | Size | Reason |
|------|------|--------|
| `resistive_mhd.py` | 11.8 KB | Replaced by `jax_frc/models/resistive_mhd.py` |
| `extended_mhd.py` | 14.6 KB | Replaced by `jax_frc/models/extended_mhd.py` |
| `hybrid_kinetic.py` | 18.3 KB | Replaced by `jax_frc/models/hybrid_kinetic.py` |
| `examples.py` | 6.9 KB | Uses legacy files |
| `test_simulations.py` | 8.8 KB | Tests legacy implementations |

### 2. Move `physics_utils.py` to Package

Move `physics_utils.py` → `jax_frc/physics.py`

Contains 28 plasma physics utility functions:
- Plasma parameters: `compute_alfven_speed`, `compute_plasma_frequency`, `compute_beta`, `compute_sound_speed`
- Length scales: `compute_larmor_radius`, `compute_skin_depth`, `compute_debye_length`
- Dimensionless numbers: `compute_reynolds_number`, `compute_lundquist_number`, `compute_mach_number`
- FRC-specific: `compute_frc_separatrix_radius`, `compute_frc_length`, `compute_frc_volume`, `compute_frc_beta`
- Field operations: `compute_gradient`, `compute_divergence`, `compute_curl`, `compute_laplacian`
- Energy: `compute_energy_magnetic`, `compute_energy_kinetic`, `compute_energy_thermal`, `compute_total_energy`

Update `jax_frc/__init__.py` to export the new module.

### 3. Delete Tests Using Legacy Imports

| File | Reason |
|------|--------|
| `tests/test_resistive_mhd.py` | Imports from root `resistive_mhd.py` |
| `tests/test_extended_mhd.py` | Imports from root `extended_mhd.py` |
| `tests/test_hybrid_kinetic.py` | Imports from root `hybrid_kinetic.py` |

### 4. Consolidate Test Directory

Move `jax_frc/tests/test_simulation.py` → `tests/test_simulation_integration.py`

Contains 6 test classes:
- `TestSimulationIntegration`
- `TestResistiveMHD`
- `TestExtendedMHD`
- `TestHybridKinetic`
- `TestEnergyConservation`
- `TestEquilibriumSolvers`

Delete `jax_frc/tests/` directory.

### Final Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── invariants/                    # Physics invariant helpers
│   ├── boundedness.py
│   ├── conservation.py
│   └── consistency.py
├── test_simulation_integration.py # Moved from jax_frc/tests/
├── test_boundaries.py
├── test_configurations.py
├── test_coupled.py
├── test_energy.py
└── ... (30+ other test files)
```

## Verification

1. Run `py -m pytest tests/ -v` before changes to establish baseline
2. Apply changes
3. Run tests again to confirm no regressions
4. Check for any remaining imports of deleted modules

## Risk Assessment

**Low risk** - The class-based implementations in `jax_frc/models/` are the actively developed versions; deleted files are legacy standalone implementations.
