# Implementation Plan: HLLD Riemann Solver and Finite Volume Method

## Overview

This plan details the implementation of a proper finite volume MHD solver for JAX-FRC,
including the HLLD Riemann solver and coupled evolution of all MHD variables.

## Background

### Current State
- JAX-FRC uses separate evolution of fields (B, n, v, p)
- CT (Constrained Transport) scheme for magnetic field advection
- Central differences for other terms
- 23-53% L2 errors compared to AGATE reference

### Target State
- Full finite volume method with coupled MHD evolution
- HLLD Riemann solver (5-wave structure)
- <10% L2 errors compared to AGATE

## Architecture

### New Module Structure
```
jax_frc/solvers/
├── riemann/
│   ├── __init__.py          # Module exports
│   ├── wave_speeds.py       # MHD wave speeds (DONE)
│   ├── reconstruction.py    # PLM/PPM with limiters (DONE)
│   ├── hll.py              # HLL solver (DONE, needs fixes)
│   ├── hlld.py             # HLLD solver (NEW)
├── finite_volume/
│   ├── __init__.py
│   ├── mhd_system.py       # Full MHD conserved variables
│   ├── flux.py             # Physical flux computation
│   └── update.py           # Finite volume update
```

## Phase 1: Fix HLL Solver (1-2 days)

### Problem
Current HLL solver is unstable because it only evolves B field while
other fields (n, v, p) are evolved separately.

### Solution
1. Implement full MHD conserved variable system
2. Compute HLL flux for all 8 variables (rho, mom_x/y/z, E, Bx/y/z)
3. Update all variables consistently

### Tasks
- [ ] Create `MHDConserved` dataclass for 8-variable state
- [ ] Implement `conserved_to_primitive()` and `primitive_to_conserved()`
- [ ] Fix `hll_flux_3d()` to return full state update
- [ ] Add proper CFL condition based on fast magnetosonic speed
- [ ] Test with Brio-Wu shock tube

## Phase 2: HLLD Riemann Solver (3-4 days)

### Algorithm (Miyoshi & Kusano 2005)
HLLD approximates the Riemann fan with 5 waves:
```
     S_L    S_L*   S_M    S_R*   S_R
      |      |      |      |      |
  U_L | U_L* | U_L**| U_R**| U_R* | U_R
      |      |      |      |      |
    Fast   Alfven Contact Alfven Fast
```

### Tasks
- [ ] Implement wave speed estimates (Davis or Roe-averaged)
- [ ] Compute contact wave speed S_M from jump conditions
- [ ] Compute total pressure in star region p_T*
- [ ] Compute U_L*, U_R* intermediate states
- [ ] Compute Alfven wave speeds S_L*, S_R*
- [ ] Compute U_L**, U_R** double-star states
- [ ] Return flux based on wave position
- [ ] Handle degenerate cases (Bn ~ 0, vacuum)

### Key Equations
```python
# Contact wave speed
S_M = ((S_R - v_Rn)*rho_R*v_Rn - (S_L - v_Ln)*rho_L*v_Ln
       - p_R + p_L) / ((S_R - v_Rn)*rho_R - (S_L - v_Ln)*rho_L)

# Total pressure in star region
p_T* = p_L + rho_L*(S_L - v_Ln)*(S_M - v_Ln)

# Star state density
rho_L* = rho_L * (S_L - v_Ln) / (S_L - S_M)
```

## Phase 3: Finite Volume Framework (2-3 days)

### Tasks
- [ ] Create `FiniteVolumeMHD` model class
- [ ] Implement dimension-by-dimension flux computation
- [ ] Add source terms (gravity, resistivity, etc.)
- [ ] Implement adaptive time stepping with CFL
- [ ] Add flux limiters for stability

### Interface
```python
@dataclass(frozen=True)
class FiniteVolumeMHD(PhysicsModel):
    """Full finite volume MHD solver.

    Args:
        gamma: Adiabatic index (default 5/3)
        riemann_solver: "hll", "hlld" (default), or "hlle"
        reconstruction: "plm" (default), "ppm", or "weno5"
        limiter: "mc" (default), "minmod", "superbee"
        limiter_beta: MC limiter parameter (default 1.3)
        ct_method: "gardiner_stone" (default) or "upwind"
        eta: Resistivity (default 0)
    """
    gamma: float = 5/3
    riemann_solver: str = "hlld"
    reconstruction: str = "plm"
    limiter: str = "mc"
    limiter_beta: float = 1.3
    ct_method: str = "gardiner_stone"
    eta: float = 0.0
```

## Phase 4: Testing and Validation (2-3 days)

### Unit Tests
- [ ] Wave speed calculations
- [ ] HLLD flux consistency (F(U,U) = physical flux)
- [ ] HLLD symmetry (F(U_L, U_R) = -F(U_R, U_L))
- [ ] Contact discontinuity resolution
- [ ] Alfven wave resolution

### Integration Tests
- [ ] Brio-Wu shock tube (1D)
- [ ] Orszag-Tang vortex (2D)
- [ ] MHD rotor (2D)
- [ ] Blast wave (2D)

### Validation
- [ ] Compare with AGATE reference data
- [ ] Target: L2 error < 10%
- [ ] Verify div(B) < 1e-10

## Phase 5: Performance Optimization (1-2 days)

### Tasks
- [ ] Profile hot paths
- [ ] Optimize JAX vmap usage
- [ ] Consider GPU acceleration
- [ ] Benchmark vs CT scheme

## Risk Mitigation

### Risk 1: Numerical Instability
- Add density/pressure floors
- Fall back to HLL for degenerate cases
- Use entropy fix for transonic rarefactions

### Risk 2: div(B) Errors
- Monitor div(B) in tests
- Use Gardiner-Stone EMF averaging
- Compare with spectral CT as reference

### Risk 3: Performance
- Profile early and often
- Use JAX primitives (lax.scan, vmap)
- Consider mixed precision

## References

1. Miyoshi & Kusano (2005) "A multi-state HLL approximate Riemann solver..."
2. Gardiner & Stone (2005) "An unsplit Godunov method for ideal MHD..."
3. Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
4. Stone et al. (2008) "Athena: A New Code for Astrophysical MHD"
5. AGATE source code: agate/agate/baseRiemann.py, slopeBuilder.py

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Fix HLL | 1-2 days | None |
| 2. HLLD | 3-4 days | Phase 1 |
| 3. FV Framework | 2-3 days | Phase 2 |
| 4. Testing | 2-3 days | Phase 3 |
| 5. Optimization | 1-2 days | Phase 4 |

**Total: ~12-17 days**

## Success Criteria

- [ ] HLLD solver passes all unit tests
- [ ] Brio-Wu shock tube matches reference (L2 < 5%)
- [ ] Orszag-Tang validation error < 10% vs AGATE
- [ ] div(B) remains < 1e-10 throughout simulation
- [ ] No NaN/Inf in any test case
- [ ] Performance overhead < 3x vs CT scheme
