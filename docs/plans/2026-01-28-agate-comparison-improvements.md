# JAX-FRC Improvements Based on AGATE Reference Implementation

**Date:** 2026-01-28
**Status:** Proposed
**Author:** Claude Code Analysis
**Reference:** AGATE (Accelerated Grid-Agnostic Techniques for Astrophysics)

## Executive Summary

This document presents recommendations for improving JAX-FRC based on a comprehensive comparison with AGATE, a NASA-developed finite-volume MHD framework. While JAX-FRC has excellent foundations for FRC fusion reactor simulation, several gaps exist in models, solvers, and validation infrastructure that should be addressed for production readiness.

---

## 1. Physics Models Analysis

### 1.1 Current JAX-FRC Models

| Model | File | Status | Key Features |
|-------|------|--------|--------------|
| ResistiveMHD | `models/resistive_mhd.py` | ✅ Good | Direct B-field evolution, resistive diffusion |
| ExtendedMHD | `models/extended_mhd.py` | ⚠️ Needs work | Hall term, electron pressure, halo model |
| HybridKinetic | `models/hybrid_kinetic.py` | ✅ Good | Delta-f PIC, Boris pusher, CIC deposition |
| NeutralFluid | `models/neutral_fluid.py` | ⚠️ Needs work | HLLE Riemann solver, 1st order only |
| BurningPlasma | `models/burning_plasma.py` | ⚠️ Needs work | Multi-fuel burn, circuit coupling incomplete |

### 1.2 Critical Model Issues

#### Issue 1: Boundary Conditions Not Integrated
- **Location:** `jax_frc/operators.py:261-467`
- **Problem:** `Geometry` has `bc_x`, `bc_y`, `bc_z` fields but operators always use periodic boundaries via `jnp.roll`
- **Impact:** Cannot simulate realistic FRC geometries with walls

**Current code:**
```python
# operators.py line 273-276
df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * geometry.dx)
```

**Recommended fix:**
```python
def gradient_3d(f: Array, geometry: Geometry) -> Array:
    df_dx = _gradient_with_bc(f, geometry.dx, axis=0, bc=geometry.bc_x)
    df_dy = _gradient_with_bc(f, geometry.dy, axis=1, bc=geometry.bc_y)
    df_dz = _gradient_with_bc(f, geometry.dz, axis=2, bc=geometry.bc_z)
    return jnp.stack([df_dx, df_dy, df_dz], axis=-1)

def _gradient_with_bc(f, dx, axis, bc):
    if bc == "periodic":
        return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dx)
    elif bc == "dirichlet":
        # One-sided differences at boundaries
        ...
    elif bc == "neumann":
        # Zero gradient at boundaries
        ...
```

#### Issue 2: No div(B)=0 Enforcement in MHD Models
- **Location:** `models/resistive_mhd.py:60-65`
- **Problem:** `apply_constraints()` is a no-op
- **Impact:** Divergence errors accumulate over time

**Recommended fix:**
```python
def apply_constraints(self, state: State, geometry: Geometry) -> State:
    from jax_frc.solvers.divergence_cleaning import clean_divergence
    B_clean = clean_divergence(state.B, geometry)
    return state.replace(B=B_clean)
```

#### Issue 3: Semi-Implicit Solver is Not Truly Implicit
- **Location:** `solvers/semi_implicit.py:80-120`
- **Problem:** Uses damped explicit instead of proper CG solve
- **Impact:** Still has CFL constraint from Hall term

**Current (damped explicit):**
```python
dB_implicit = dB_explicit / (1 + dt**2 * damping_factor)
```

**Recommended (proper implicit):**
```python
def hall_operator(B):
    return B - dt**2 * L_hall(B)

B_new, info = conjugate_gradient(hall_operator, rhs, tol=1e-6)
```

### 1.3 AGATE Patterns to Adopt

#### Pattern 1: State Hierarchy with Pointer Views

---

## 2. Solver Improvements

### 2.1 Current Solver Status

| Solver | File | Order | Status |
|--------|------|-------|--------|
| EulerSolver | `solvers/explicit.py` | 1st | ✅ Working |
| RK4Solver | `solvers/explicit.py` | 4th | ⚠️ Only updates B in stages |
| SemiImplicitSolver | `solvers/semi_implicit.py` | 1st | ❌ Not truly implicit |
| ImexSolver | `solvers/imex.py` | 2nd | ✅ Good |
| TimeController | `solvers/time_controller.py` | N/A | ⚠️ Basic |

### 2.2 Missing Solvers from AGATE

#### Add RK2 Solver (Heun's Method)
```python
class RK2Solver(Solver):
    """2nd-order Runge-Kutta (Heun's method)"""

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        k1 = model.compute_rhs(state, geometry)
        state_mid = state.replace(B=state.B + dt * k1.B)
        k2 = model.compute_rhs(state_mid, geometry)

        new_B = state.B + dt/2 * (k1.B + k2.B)
        return state.replace(B=new_B)
```

#### Add RK3 SSP Solver (Strong Stability Preserving)
```python
class RK3SSPSolver(Solver):
    """3rd-order Strong Stability Preserving RK"""

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # Stage 1
        k1 = model.compute_rhs(state, geometry)
        u1 = state.replace(B=state.B + dt * k1.B)

        # Stage 2
        k2 = model.compute_rhs(u1, geometry)
        u2 = state.replace(B=0.75*state.B + 0.25*(u1.B + dt*k2.B))

        # Stage 3
        k3 = model.compute_rhs(u2, geometry)
        new_B = (1/3)*state.B + (2/3)*(u2.B + dt*k3.B)

        return state.replace(B=new_B)
```

### 2.3 Add MC Limiter for Reconstruction

**Location:** New file `jax_frc/solvers/reconstruction.py`

```python
@jax.jit
def mc_limiter(U_L: Array, U_C: Array, U_R: Array, beta: float = 1.3) -> Array:
    """Monotonized Central limiter for 2nd-order reconstruction.

    Args:
        U_L: Left cell values
        U_C: Center cell values
        U_R: Right cell values
        beta: Limiter parameter (1.0-2.0, default 1.3)

    Returns:
        Limited slopes for reconstruction
    """
    slope_L = beta * (U_C - U_L) / 2.0
    slope_R = beta * (U_R - U_C) / 2.0
    slope_C = (U_R - U_L) / 4.0

    both_neg = (slope_L < 0) & (slope_R < 0)
    both_pos = (slope_L > 0) & (slope_R > 0)

    return jnp.where(
        both_neg,
        jnp.maximum(jnp.maximum(slope_L, slope_R), slope_C),
        jnp.where(
            both_pos,
            jnp.minimum(jnp.minimum(slope_L, slope_R), slope_C),
            0.0
        )
    )
```

### 2.4 Fix Divergence Cleaning Convergence

**Location:** `jax_frc/solvers/divergence_cleaning.py`

**Current problem:** Fixed Jacobi iterations with no convergence check

**Recommended fix:**
```python
def clean_divergence(B: Array, geometry: Geometry,
                     tol: float = 1e-6, max_iter: int = 100) -> tuple[Array, float]:
    """Clean divergence using projection method with CG solver.

    Returns:
        Tuple of (cleaned B field, final residual)
    """
    div_B = divergence_3d(B, geometry)

    def laplacian_op(phi):
        return laplacian_3d(phi, geometry)

    phi, info = conjugate_gradient(laplacian_op, div_B, tol=tol, max_iter=max_iter)

    grad_phi = gradient_3d(phi, geometry)
    B_clean = B - grad_phi

    return B_clean, info.residual
```

---

## 3. Test Case and Validation Improvements

### 3.1 Current Test Coverage

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | ~600 | ✅ Good coverage |
| Invariant tests | 15+ | ✅ Excellent framework |
| Analytic validation | 1 (diffusion) | ⚠️ Limited |
| Standard benchmarks | 0 | ❌ Missing |
| FRC-specific | 3 (Belova) | ⚠️ Qualitative only |

### 3.2 Missing Standard Benchmarks

#### Orszag-Tang Vortex (2D MHD Turbulence)
**File:** `validation/cases/mhd/orszag_tang.py`

```python
"""Orszag-Tang Vortex Validation

2D MHD turbulence test with shock interactions.

Reference:
    Orszag & Tang, J. Fluid Mech. 90, 129 (1979)
"""
import jax.numpy as jnp
from math import pi

def setup_configuration() -> dict:
    return {
        'name': 'orszag_tang',
        'geometry': {
            'nx': 256, 'ny': 256, 'nz': 4,
            'x_min': 0.0, 'x_max': 2*pi,
            'y_min': 0.0, 'y_max': 2*pi,
            'bc_x': 'periodic', 'bc_y': 'periodic', 'bc_z': 'periodic'
        },
        'initial_conditions': {
            'rho': 25/(36*pi),
            'p': 5/(12*pi),
            'vx': lambda x, y: -jnp.sin(y),
            'vy': lambda x, y: jnp.sin(x),
            'Bx': lambda x, y: -jnp.sin(y),
            'By': lambda x, y: jnp.sin(2*x)
        },
        'physics': {
            'gamma': 5/3
        },
        'solver': {
            'type': 'rk4',
            'cfl': 0.4
        },
        'time': {
            't_end': 0.5
        },
        'validation': {
            'type': 'reference_data',
            'source': 'agate_hdf5',
            'resolutions': [256, 512, 1024],
            'metrics': ['kinetic_energy', 'magnetic_energy', 'enstrophy']
        }
    }
```

### 3.3 AGATE HDF5 Reference Data Integration

**File:** `jax_frc/validation/agate_loader.py`

```python
"""Load AGATE reference data from Zenodo HDF5 files."""
import h5py
from pathlib import Path

AGATE_DATASETS = {
    'hall_gem_512': 'https://zenodo.org/record/XXXXX/files/hall_gem_512.h5',
    'orszag_tang_256': 'https://zenodo.org/record/XXXXX/files/ot_256.h5',
    'orszag_tang_512': 'https://zenodo.org/record/XXXXX/files/ot_512.h5',
}

class AgateReferenceLoader:
    def __init__(self, cache_dir: Path = Path('validation/references/agate')):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_snapshot(self, dataset: str, time_index: int) -> dict:
        """Load grid and state from AGATE HDF5 file."""
        path = self._ensure_downloaded(dataset)
        with h5py.File(path, 'r') as f:
            return {
                'grid': {
                    'x': f['grid/x'][:],
                    'y': f['grid/y'][:],
                    'z': f['grid/z'][:]
                },
                'state': {
                    'rho': f[f'snapshots/{time_index}/rho'][:],
                    'vx': f[f'snapshots/{time_index}/vx'][:],
                    'vy': f[f'snapshots/{time_index}/vy'][:],
                    'Bx': f[f'snapshots/{time_index}/Bx'][:],
                    'By': f[f'snapshots/{time_index}/By'][:],
                    'p': f[f'snapshots/{time_index}/p'][:]
                }
            }
```

### 3.4 Quantitative FRC Metrics

**Enhance Belova cases with quantitative metrics:**

```python
# validation/cases/frc/metrics.py

def compute_frc_metrics(state: State, geometry: Geometry) -> dict:
    """Compute quantitative FRC metrics for validation."""
    return {
        'separatrix_radius': _find_separatrix_radius(state.B, geometry),
        'elongation': _compute_elongation(state.B, geometry),
        'trapped_flux': _compute_trapped_flux(state.B, geometry),
        'beta_separatrix': _compute_beta_at_separatrix(state, geometry),
        'kinetic_parameter': _compute_s_star(state, geometry),
        'magnetic_energy': _compute_magnetic_energy(state.B, geometry),
        'thermal_energy': _compute_thermal_energy(state.p, geometry),
        'kinetic_energy': _compute_kinetic_energy(state, geometry)
    }

def _find_separatrix_radius(B: Array, geometry: Geometry) -> float:
    """Find radius where B_z changes sign (separatrix)."""
    # Find zero-crossing of B_z at midplane
    midplane_idx = geometry.nz // 2
    B_z_midplane = B[:, :, midplane_idx, 2]
    # ... interpolation to find exact radius
```

---

## 4. Invariant Testing Enhancements

### 4.1 New Invariants to Add

```python
# tests/invariants/mhd.py

class DivergenceTracking(Invariant):
    """Track div(B) error over time."""

    def __init__(self, max_growth_factor: float = 10.0):
        self.max_growth_factor = max_growth_factor
        self.initial_div_B = None

    def check(self, state_before: State, state_after: State,
              geometry: Geometry) -> InvariantResult:
        div_B = divergence_3d(state_after.B, geometry)
        div_B_norm = jnp.linalg.norm(div_B)

        if self.initial_div_B is None:
            self.initial_div_B = div_B_norm

        growth = div_B_norm / (self.initial_div_B + 1e-15)
        passed = growth < self.max_growth_factor

        return InvariantResult(
            passed=passed,
            name='DivergenceTracking',
            value=float(growth),
            tolerance=self.max_growth_factor,
            message=f'div(B) growth factor: {growth:.2f}'
        )


class HelicityConservation(Invariant):
    """Check magnetic helicity conservation."""

    def __init__(self, rtol: float = 0.01):
        self.rtol = rtol

    def check(self, state_before: State, state_after: State,
              geometry: Geometry) -> InvariantResult:
        H_before = self._compute_helicity(state_before.B, geometry)
        H_after = self._compute_helicity(state_after.B, geometry)

        rel_change = abs(H_after - H_before) / (abs(H_before) + 1e-15)
        passed = rel_change < self.rtol

        return InvariantResult(
            passed=passed,
            name='HelicityConservation',
            value=float(rel_change),
            tolerance=self.rtol,
            message=f'Helicity change: {rel_change*100:.2f}%'
        )

    def _compute_helicity(self, B: Array, geometry: Geometry) -> float:
        """H = ∫ A·B dV where B = curl(A)"""
        # Compute vector potential A from B
        # ...
```

---

## 5. Implementation Roadmap

### Phase 1: Critical Fixes (1-2 weeks)
| Task | Priority | Effort | Files |
|------|----------|--------|-------|
| Fix BC integration in operators | P0 | 3 days | `operators.py` |
| Integrate div(B) cleaning | P0 | 1 day | `resistive_mhd.py`, `extended_mhd.py` |
| Add RK2 solver | P1 | 1 day | `explicit.py` |
| Fix RK4 to evolve all fields | P1 | 1 day | `explicit.py` |

### Phase 2: Solver Improvements (2-3 weeks)
| Task | Priority | Effort | Files |
|------|----------|--------|-------|
| Fix semi-implicit (proper CG) | P0 | 3 days | `semi_implicit.py` |
| Add MC limiter | P1 | 2 days | New `reconstruction.py` |
| Fix div cleaning convergence | P1 | 1 day | `divergence_cleaning.py` |
| Add RK3 SSP solver | P2 | 1 day | `explicit.py` |

### Phase 3: Validation Suite (2-3 weeks)
| Task | Priority | Effort | Files |
|------|----------|--------|-------|
| Implement Brio-Wu | P0 | 2 days | `validation/cases/mhd/` |
| Implement Orszag-Tang | P0 | 2 days | `validation/cases/mhd/` |
| AGATE HDF5 loader | P1 | 2 days | `validation/agate_loader.py` |
| Quantitative FRC metrics | P1 | 3 days | `validation/cases/frc/` |

### Phase 4: Production Features (3-4 weeks)
| Task | Priority | Effort | Files |
|------|----------|--------|-------|
| Regression baselines | P1 | 2 days | `validation/` |
| New invariants | P2 | 2 days | `tests/invariants/` |
| Performance benchmarks | P2 | 2 days | `benchmarks/` |

---

## 6. Summary

### Key Strengths of JAX-FRC
- Excellent JAX patterns (immutable state, JIT, pytrees)
- FRC-specific physics (Belova cases, burning plasma)
- Comprehensive invariant testing framework
- Clean protocol-based architecture

### Critical Gaps to Address
1. **Boundary conditions** - Operators ignore BC settings
2. **Semi-implicit solver** - Not truly implicit
3. **Standard benchmarks** - Missing Brio-Wu, Orszag-Tang
4. **div(B) enforcement** - Not integrated into models

### AGATE Patterns to Adopt
1. Modular component composition (Evolver pattern)
2. MC limiter for reconstruction
3. RK2/RK3 time steppers
4. HDF5 reference data for validation

### Expected Outcomes
- Production-ready boundary condition handling
- 10-100× larger timesteps with proper semi-implicit
- Validated against standard MHD benchmarks
- Quantitative FRC metrics for reactor design
