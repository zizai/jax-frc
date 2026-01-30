# HLLD Finite Volume MHD Implementation

## Overview

This document summarizes the implementation of a proper finite volume MHD solver for JAX-FRC, including the HLLD Riemann solver and coupled evolution of all MHD variables.

### Goal
Replace the separate field evolution approach with a full finite volume method using HLLD Riemann solver to reduce L2 errors from 23-53% to <10% compared to AGATE reference.

---

## Completed Work

### Phase 1: Full Finite Volume MHD with Coupled Evolution ✅

**Commit:** `bbe4f6e`

Created the foundation for finite volume MHD:

- **`mhd_state.py`**: MHD conserved/primitive state types
  - `MHDConserved`: 8-variable state (rho, mom_x/y/z, E, Bx/y/z)
  - `MHDPrimitive`: Primitive variables (rho, vx/y/z, p, Bx/y/z)
  - Conversion functions between conserved and primitive
  - Physical flux computation for all directions

- **`hll_full.py`**: Full HLL solver for 8-variable MHD system
  - Coupled evolution of all MHD variables
  - Davis wave speed estimates
  - Proper CFL condition based on fast magnetosonic speed

- **`finite_volume_mhd.py`**: FiniteVolumeMHD model class
  - Supports `hll` and `hlld` solvers
  - PLM reconstruction with MC-beta limiter (beta=1.3)
  - Configurable CFL number

### Phase 2: HLLD 5-Wave Riemann Solver ✅

**Commit:** `86fc24e`

Implemented the Miyoshi & Kusano (2005) HLLD algorithm:

- **`hlld.py`**: HLLD 5-wave solver
  - Fast magnetosonic waves (S_L, S_R)
  - Alfven waves (S_L*, S_R*)
  - Contact discontinuity (S_M)
  - Star and double-star intermediate states
  - Proper handling of degenerate cases (Bn ~ 0)

## Test Results

### Orszag-Tang Vortex (256x256, t=0.5)

| Solver | CFL | Stable | max|div(B)| | Density Range |
|--------|-----|--------|-------------|---------------|
| HLLD | 0.3 | ✅ | 19 | [0.59, 2.19] |
| HLLD | 0.4 | ✅ | 13 | [0.58, 2.31] |

### Unit Tests
- 641 passed, 4 failed (pre-existing failures unrelated to HLLD)

---

## Remaining Work

### Phase 4: Source Terms (Optional)

Add resistivity and other source terms to the finite volume framework.

**Tasks:**
- [ ] Add resistivity term to energy equation
- [ ] Implement operator splitting for stiff source terms
- [ ] Add gravity source term (if needed)
- [ ] Test with resistive MHD problems

**Files to modify:**
- `jax_frc/models/finite_volume_mhd.py`

**Estimated effort:** 1-2 days

### Phase 5: Testing and Validation

Comprehensive testing against AGATE reference data.

**Unit Tests:**
- [ ] Wave speed calculations
- [ ] HLLD flux consistency: F(U,U) = physical flux
- [ ] HLLD symmetry: F(U_L, U_R) = -F(U_R, U_L)
- [ ] Contact discontinuity resolution
- [ ] Alfven wave resolution

**Integration Tests:**
- [ ] Brio-Wu shock tube (1D)
- [ ] Orszag-Tang vortex (2D)
- [ ] MHD rotor (2D)
- [ ] Blast wave (2D)

**Validation:**
- [ ] Compare with AGATE reference data
- [ ] Target: L2 error < 10%
- [ ] Document div(B) behavior

**Files to create:**
- `tests/test_hlld_solver.py`
- `tests/test_finite_volume_mhd.py`

**Estimated effort:** 2-3 days

### Phase 6: Performance Optimization

Optimize for GPU acceleration and reduce compilation time.

**Tasks:**
- [ ] Profile hot paths (reconstruction, flux computation)
- [ ] Optimize JAX vmap usage
- [ ] Reduce JIT compilation time
- [ ] Benchmark vs existing CT scheme
- [ ] Consider mixed precision for performance

**Performance targets:**
- Overhead < 3x vs CT scheme
- JIT compilation < 30s

**Estimated effort:** 1-2 days

---

## Known Issues

### div(B) Growth
The HLLD solver without proper CT causes div(B) to grow over time. This is expected behavior for finite volume MHD on a cell-centered grid. Options:
1. Accept div(B) growth (current approach)
2. Implement true CT on staggered grid (significant refactor)
3. Use hyperbolic divergence cleaning (GLM method)

### Divergence Cleaning Instability
The projection-based divergence cleaning modifies B without adjusting energy, causing inconsistency. The GLM (Generalized Lagrange Multiplier) method may be more stable.

---

## File Summary

```
jax_frc/solvers/riemann/
├── __init__.py          # Module exports (updated)
├── wave_speeds.py       # MHD wave speeds
├── reconstruction.py    # PLM/PPM with limiters
├── mhd_state.py         # MHD state types (NEW)
├── hll.py               # Original HLL (B-field only)
├── hll_full.py          # Full HLL solver (NEW)
├── hlld.py              # HLLD 5-wave solver (NEW)
└── ct_hlld.py           # CT utilities (unused)

jax_frc/models/
└── finite_volume_mhd.py # FiniteVolumeMHD model (NEW)
```

---

## References

1. Miyoshi & Kusano (2005) "A multi-state HLL approximate Riemann solver for ideal MHD"
2. Gardiner & Stone (2005) "An unsplit Godunov method for ideal MHD via CT"
3. Stone et al. (2008) "Athena: A New Code for Astrophysical MHD"
4. AGATE source: `agate/agate/baseRiemann.py`, `slopeBuilder.py`
