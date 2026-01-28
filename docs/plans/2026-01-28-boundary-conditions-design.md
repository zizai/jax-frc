# Boundary Condition Improvements for MHD Operators

**Date:** 2026-01-28
**Status:** Design Complete
**Author:** Claude Opus 4.5

## Overview

Extend jax-frc operators to support non-periodic boundary conditions for full MHD simulations, and validate with magnetic diffusion test cases using open and Dirichlet boundaries.

## Goals

1. Add Dirichlet/Neumann BC support to `curl_3d` and `divergence_3d` operators
2. Integrate boundary E-field handling into Constrained Transport (CT) solver
3. Validate with magnetic diffusion using open (Neumann) boundaries
4. Add separate Dirichlet BC test with conducting wall boundaries

## Background

### Current State

- `gradient_3d` and `laplacian_3d` support periodic/Dirichlet/Neumann BCs
- `curl_3d` and `divergence_3d` use `jnp.roll()` (periodic only)
- Magnetic diffusion validation exists but uses periodic BCs only
- CT solver preserves ∇·B=0 in interior but lacks boundary handling

### Key Constraint

MHD boundaries must preserve ∇·B=0. The boundary condition guide specifies:
- **Ghost cell method** for finite difference operators
- **CT scheme** handles ∇·B=0 at boundaries via electric field (E), not B directly
- **Open/Outflow**: Zero-gradient extrapolation (`Q_ghost = Q_interior`)
- **Conducting walls**: Reflect normal components, symmetric tangential

## Design

### 1. Operator Boundary Condition Extensions

Extend `curl_3d` and `divergence_3d` in `jax_frc/operators.py` using ghost cell method.

#### Ghost Cell Strategy

For a field `F` at boundary with `n_ghost=1`:

| BC Type | Ghost Cell Value | Physical Meaning |
|---------|------------------|------------------|
| **Periodic** | `F[0] = F[-2]`, `F[-1] = F[1]` | Wrap around |
| **Dirichlet** | `F[0] = -F[1]` (reflect + negate) | F=0 at boundary face |
| **Neumann** | `F[0] = F[1]` (copy) | ∂F/∂n=0 at boundary |

#### Implementation

```python
def _pad_with_bc(F: jnp.ndarray, axis: int, bc: str) -> jnp.ndarray:
    """Pad field with ghost cells based on BC type."""
    if bc == "periodic":
        return jnp.roll(...)  # existing behavior
    elif bc == "dirichlet":
        # Reflect and negate for F=0 at boundary
        return _reflect_pad(F, axis, negate=True)
    elif bc == "neumann":
        # Copy for zero gradient
        return _reflect_pad(F, axis, negate=False)
```

The `curl_3d` and `divergence_3d` functions will:
1. Use `_pad_with_bc` before computing finite differences
2. Strip ghost cells from the result

### 2. CT Boundary Integration

Extend Constrained Transport solver to handle boundary electric fields.

#### Boundary E-Field Handling

For open boundaries, apply zero-gradient extrapolation to E before computing curl:

```python
def _apply_boundary_E(E: jnp.ndarray, geometry: Geometry) -> jnp.ndarray:
    """Apply boundary conditions to electric field for CT scheme."""
    E_bc = E.copy()

    for axis, bc in enumerate([geometry.bc_x, geometry.bc_y, geometry.bc_z]):
        if bc == "periodic":
            continue  # Handled by spectral method
        elif bc in ("neumann", "dirichlet"):
            # Zero-gradient: E_ghost = E_interior
            E_bc = _extrapolate_boundary(E_bc, axis)

    return E_bc
```

#### Integration Point

In `CTSolver.step()`:
1. Compute E from physics model (E = -v×B + ηJ)
2. Apply boundary conditions to E via `_apply_boundary_E()`
3. Compute `∂B/∂t = -∇×E` using BC-aware curl
4. Update B

This ensures ∇·B=0 is preserved at boundaries because we control flux through boundary faces via E.

### 3. Magnetic Diffusion Validation - Open BC Test

Modify existing magnetic diffusion validation to use open (Neumann) boundaries.

#### Configuration Changes

```python
@dataclass
class MagneticDiffusionConfiguration:
    bc_x: str = "neumann"  # Changed from "periodic"
    bc_y: str = "neumann"  # Changed from "periodic"
    bc_z: str = "neumann"  # Pseudo-dimension
```

#### Domain Sizing

- Initial Gaussian width: σ = 0.1
- Domain extent: L = 1.0 (gives [-1, 1] domain)
- At boundary: `exp(-1²/(2×0.1²)) = exp(-50) ≈ 0`

Domain is large enough - Gaussian decays to effectively zero before boundaries.

#### Analytic Solution

Infinite domain Gaussian (unchanged):

```python
def analytic_solution(x, y, t, B_peak, sigma, eta):
    sigma_t_sq = sigma**2 + 2 * eta * t
    return B_peak * (sigma**2 / sigma_t_sq) * np.exp(-(x**2 + y**2) / (2 * sigma_t_sq))
```

### 4. Magnetic Diffusion Validation - Dirichlet BC Test

Add new validation test for conducting wall boundaries (B=0 at walls).

#### Analytic Solution

Multi-mode Fourier series for domain [0, L] × [0, L] with B=0 at boundaries:

```python
def analytic_dirichlet(x, y, t, L, eta, modes):
    """Multi-mode Fourier solution for diffusion with Dirichlet BCs.

    B(x,y,t) = Σ Aₙₘ sin(nπx/L) sin(mπy/L) exp(-λₙₘ·η·t)
    where λₙₘ = π²(n² + m²) / L²
    """
    B = 0.0
    for n, m, A in modes:
        lambda_nm = np.pi**2 * (n**2 + m**2) / L**2
        B += A * np.sin(n*np.pi*x/L) * np.sin(m*np.pi*y/L) * np.exp(-lambda_nm * eta * t)
    return B
```

#### Initial Condition (Multi-mode)

```python
modes = [
    (1, 1, 1.0),   # Fundamental mode
    (2, 1, 0.3),   # First harmonic in x
    (1, 2, 0.3),   # First harmonic in y
    (2, 2, 0.1),   # Diagonal harmonic
]
```

#### Validation Metrics

1. **Mode decay rates**: Each mode decays as `exp(-λₙₘ·η·t)`
   - λ₁₁ = 2π²/L² (slowest)
   - λ₂₁ = λ₁₂ = 5π²/L² (faster)
   - λ₂₂ = 8π²/L² (fastest)

2. **L2 error**: < 5% between numerical and analytic solution

3. **Boundary enforcement**: B ≈ 0 at all boundaries throughout simulation

## File Changes

### Files to Modify

| File | Changes |
|------|---------|
| `jax_frc/operators.py` | Add `_pad_with_bc()`, `_reflect_pad()`, update `curl_3d()` and `divergence_3d()` |
| `jax_frc/solvers/constrained_transport.py` | Add `_apply_boundary_E()`, `_extrapolate_boundary()`, integrate into `step()` |
| `jax_frc/configurations/magnetic_diffusion.py` | Add `domain_type` field, update BC defaults |
| `validation/cases/analytic/magnetic_diffusion.py` | Add open BC test mode |
| `tests/test_operators_3d.py` | Add curl/div BC tests |

### New Files

| File | Purpose |
|------|---------|
| `validation/cases/analytic/magnetic_diffusion_dirichlet.py` | Dirichlet BC validation case |

## Implementation Order

1. **Operators** - Add BC support to `curl_3d` and `divergence_3d`
2. **Tests** - Add operator BC tests
3. **CT Solver** - Add boundary E-field handling
4. **Validation (Open BC)** - Update magnetic diffusion config
5. **Validation (Dirichlet)** - Add new test case with Fourier solution

## Success Criteria

- [ ] `curl_3d` and `divergence_3d` pass BC tests (periodic, Dirichlet, Neumann)
- [ ] CT solver preserves ∇·B=0 at non-periodic boundaries
- [ ] Magnetic diffusion (open BC): L2 error < 5%
- [ ] Magnetic diffusion (Dirichlet BC): L2 error < 5%, all modes decay correctly

## References

- `boundary_condition_guide.md` - MHD boundary condition theory and implementation
- `validation/cases/analytic/magnetic_diffusion.md` - Existing physics documentation
