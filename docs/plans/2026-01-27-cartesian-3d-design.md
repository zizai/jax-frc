# 3D Cartesian Coordinate System Implementation

## Overview

Replace the cylindrical (r, z, φ) coordinate system with full 3D Cartesian (x, y, z) coordinates to enable simulation of non-axisymmetric phenomena including tilt modes, 3D instabilities, and asymmetric merging dynamics.

## Motivation

The current implementation is 2D axisymmetric, which cannot capture:
- FRC tilt/shift modes (n=1 azimuthal structure)
- 3D instabilities and turbulence
- Asymmetric merging dynamics
- Non-axisymmetric boundary effects

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Coordinate system | Full 3D Cartesian (x, y, z) | Simplest, most general, works for any geometry |
| Scope | Full replacement | No legacy cylindrical code to maintain |
| Boundary conditions | Flexible per-axis | Different studies need different setups |
| Field representation | Evolve B directly | Required for 3D; flux function ψ is inherently 2D |
| Validation | Diffusion + Alfvén wave | Both have analytic solutions for verification |

---

## 1. Geometry

### Current State

```python
@dataclass(frozen=True)
class Geometry:
    coord_system: Literal["cylindrical", "cartesian"]
    nr: int; nz: int
    r_min: float; r_max: float
    z_min: float; z_max: float
```

Cell volumes include $2\pi r$ factor. Enforces `r_min > 0` to avoid axis singularity.

### New 3D Cartesian Geometry

```python
@dataclass(frozen=True)
class Geometry:
    # Grid dimensions
    nx: int
    ny: int
    nz: int

    # Domain bounds
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float

    # Boundary conditions per axis
    bc_x: Literal["periodic", "dirichlet", "neumann"]
    bc_y: Literal["periodic", "dirichlet", "neumann"]
    bc_z: Literal["periodic", "dirichlet", "neumann"]
```

**Derived properties:**
- `x`, `y`, `z` - 1D coordinate arrays
- `dx`, `dy`, `dz` - uniform grid spacing
- `x_grid`, `y_grid`, `z_grid` - 3D broadcast grids via `jnp.meshgrid`
- `cell_volumes` - simply `dx * dy * dz` (no geometric factors)

**Array convention:**
- Scalars: shape `(nx, ny, nz)`
- Vectors: shape `(nx, ny, nz, 3)` with components `[Fx, Fy, Fz]` in last axis

**Key simplification:** No axis singularity handling. No `r_safe`, no L'Hôpital limits.

---

## 2. Differential Operators

### Current State

`operators.py` contains cylindrical-specific operators with 1/r terms and L'Hôpital handling:
- `laplace_star` - Grad-Shafranov operator
- `divergence_cylindrical` - includes $(1/r)\partial(r f_r)/\partial r$
- `curl_cylindrical_axisymmetric` - includes $B_\theta/r$ terms

### New 3D Cartesian Operators

**Gradient (scalar → vector):**

$$
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right)
$$

```python
def gradient_3d(f, dx, dy, dz, bc_x, bc_y, bc_z):
    """Returns array of shape (nx, ny, nz, 3)."""
```

**Divergence (vector → scalar):**

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

```python
def divergence_3d(F, dx, dy, dz, bc_x, bc_y, bc_z):
    """Returns array of shape (nx, ny, nz)."""
```

**Curl (vector → vector):**

$$
\nabla \times \mathbf{F} = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}, \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}, \frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)
$$

```python
def curl_3d(F, dx, dy, dz, bc_x, bc_y, bc_z):
    """Returns array of shape (nx, ny, nz, 3)."""
```

**Laplacian (scalar → scalar):**

$$
\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
$$

```python
def laplacian_3d(f, dx, dy, dz, bc_x, bc_y, bc_z):
    """Returns array of shape (nx, ny, nz)."""
```

**Boundary condition handling:**
- Periodic: wrap with `jnp.roll`
- Dirichlet/Neumann: one-sided stencils (reuse existing code)

**Stencil accuracy:** 2nd-order central differences interior, 2nd-order one-sided at boundaries.

---

## 3. Physics Models

### Resistive MHD (Major Rewrite)

**Current:** Evolves scalar flux ψ, derives B via $B_r = -(1/r)\partial\psi/\partial z$, $B_z = (1/r)\partial\psi/\partial r$.

**New:** Evolves B directly using the induction equation:

$$
\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}
$$

$$
\mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J}, \quad \mathbf{J} = \frac{\nabla \times \mathbf{B}}{\mu_0}
$$

**State:** `ResistiveMHDState` holds `B: Array` of shape `(nx, ny, nz, 3)`.

```python
def compute_rhs(self, state, geometry):
    J = curl_3d(state.B, ...) / MU0
    E = -cross(v, state.B) + self.eta * J
    dB_dt = -curl_3d(E, ...)
    return dB_dt
```

### Extended MHD (Moderate Changes)

Already evolves B field. Update operator calls:
- `curl_cylindrical_axisymmetric` → `curl_3d`
- `divergence_cylindrical` → `divergence_3d`

Physics unchanged:

$$
\mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J} + \frac{\mathbf{J} \times \mathbf{B}}{ne} - \frac{\nabla p_e}{ne}
$$

Temperature evolution uses `laplacian_3d` for thermal conduction.

### Hybrid Kinetic (Moderate Changes)

- **Boris pusher:** Coordinate-agnostic, no changes
- **Field interpolation:** Extend from 2D bilinear to 3D trilinear
- **Current deposition:** Extend to 3D grid with shape functions
- **Equilibrium:** Replace `rigid_rotor` with generic 3D Maxwellian or user-provided function

### Neutral Fluid (Minor Changes)

Euler equations are coordinate-independent. Extend HLLE flux computation to x, y, z sweeps (dimension-by-dimension).

### Burning Plasma (No Physics Changes)

Orchestrates MHD + burn + transport. Works automatically once underlying models are 3D.

---

## 4. Solvers

### Explicit Solvers (No Changes)

`EulerSolver` and `RK4Solver` are coordinate-agnostic. They call `model.compute_rhs(state, geometry)`.

### Semi-Implicit Solver (No Changes)

Also coordinate-agnostic.

### IMEX Solver (Operator Update)

The implicit diffusion step uses conjugate gradient to solve:

$$
(I - \theta \Delta t \cdot D) \psi^{n+1} = \psi^n
$$

**Change:** Diffusion operator D becomes `laplacian_3d`. The CG solver is coordinate-agnostic.

### Divergence Cleaning (Rewrite)

Project B to satisfy $\nabla \cdot \mathbf{B} = 0$:

1. Solve Poisson equation: $\nabla^2 \phi = \nabla \cdot \mathbf{B}$
2. Correct: $\mathbf{B} \leftarrow \mathbf{B} - \nabla \phi$

```python
def clean_divergence(B, geometry):
    div_B = divergence_3d(B, ...)
    phi = poisson_solve_3d(div_B, geometry)  # Jacobi or CG
    grad_phi = gradient_3d(phi, ...)
    return B - grad_phi
```

Poisson solver uses 7-point stencil with `laplacian_3d`.

---

## 5. Equilibrium Solver

### Current Grad-Shafranov

Solves 2D axisymmetric force balance: $\Delta^* \psi = -\mu_0 r^2 p'(\psi) - F F'(\psi)$

### New 3D Force-Balance Solver

In 3D, force balance is:

$$
\mathbf{J} \times \mathbf{B} = \nabla p
$$

**Implementation:** Iterative relaxation

```python
def solve_equilibrium_3d(geometry, p_profile, B_initial, tol=1e-6):
    """Find force-balanced B field for given pressure profile."""
    B = B_initial
    for iteration in range(max_iter):
        J = curl_3d(B, ...) / MU0
        # Compute force imbalance
        F = cross(J, B) - gradient_3d(p, ...)
        # Relax B to reduce imbalance
        B = B + dt_relax * curl_3d(F, ...)
        # Clean divergence
        B = clean_divergence(B, geometry)
        if norm(F) < tol:
            break
    return B
```

**Analytic initializers:**
- `harris_sheet_3d()` - Classic reconnection setup
- `flux_rope_3d()` - Cylindrical flux rope (FRC-like)
- `uniform_field_3d()` - For wave tests

---

## 6. Diagnostics

### Volume Integrals

**Current:** $\int 2\pi r \, dr \, dz$

**New:** $\int dx \, dy \, dz$ — simply sum over cells times `dx * dy * dz`

All energy, helicity, and flux diagnostics updated accordingly.

---

## 7. Validation Test Cases

### Test 1: Magnetic Diffusion (3D Gaussian)

**Setup:** Gaussian magnetic field pulse

$$
B_z(x, y, z, t=0) = B_0 \exp\left(-\frac{x^2 + y^2 + z^2}{2\sigma_0^2}\right)
$$

**Analytic solution:**

$$
B_z(x, y, z, t) = B_0 \left(\frac{\sigma_0^2}{\sigma_0^2 + 2\eta t}\right)^{3/2} \exp\left(-\frac{x^2 + y^2 + z^2}{2(\sigma_0^2 + 2\eta t)}\right)
$$

**Validates:** `curl_3d`, `laplacian_3d`, resistive diffusion, time integration.

### Test 2: Alfvén Wave Propagation

**Setup:** Uniform background $\mathbf{B}_0 = B_0 \hat{z}$, transverse perturbation

$$
B_y(z, t=0) = \delta B \sin(k z)
$$

**Analytic solution:** Wave propagates at $v_A = B_0/\sqrt{\mu_0 \rho_0}$

$$
B_y(z, t) = \delta B \sin(k(z - v_A t))
$$

**Validates:** Ideal MHD wave physics, curl accuracy, no spurious damping.

**Convergence:** Run at multiple resolutions, verify 2nd-order convergence in L2 error.

---

## 8. Implementation Order

### Phase 1: Core Infrastructure
1. `jax_frc/core/geometry.py` - New 3D Geometry class
2. `jax_frc/operators.py` - All 3D Cartesian operators

### Phase 2: Physics Models
3. `jax_frc/models/resistive_mhd.py` - Rewrite ψ → B evolution
4. `jax_frc/models/extended_mhd.py` - Update operator calls
5. `jax_frc/models/neutral_fluid.py` - Extend to 3D
6. `jax_frc/models/hybrid_kinetic.py` - 3D interpolation, new equilibrium
7. `jax_frc/models/burning_plasma.py` - Should work once MHD updated

### Phase 3: Solvers
8. `jax_frc/solvers/imex.py` - Update diffusion operator
9. `jax_frc/solvers/divergence_cleaning.py` - 3D Poisson solve

### Phase 4: Supporting Code
10. `jax_frc/equilibrium/grad_shafranov.py` - Replace with 3D force-balance solver
11. `jax_frc/equilibrium/rigid_rotor.py` - Replace with 3D initializers
12. `jax_frc/diagnostics/` - Update volume integrals
13. `jax_frc/configurations/` - New 3D configurations

### Phase 5: Validation
14. `tests/test_operators_3d.py` - Unit tests for all operators
15. `tests/test_diffusion_3d.py` - Gaussian diffusion convergence
16. `tests/test_alfven_wave.py` - Wave propagation convergence

---

## 9. Files Summary

| Action | Files |
|--------|-------|
| **Major rewrite** | `geometry.py`, `operators.py`, `resistive_mhd.py`, `divergence_cleaning.py`, `grad_shafranov.py` |
| **Moderate update** | `extended_mhd.py`, `hybrid_kinetic.py`, `imex.py` |
| **Minor update** | `neutral_fluid.py`, `burning_plasma.py`, diagnostics, configurations |
| **New files** | `test_operators_3d.py`, `test_diffusion_3d.py`, `test_alfven_wave.py` |
| **Delete** | `rigid_rotor.py` (functionality merged into equilibrium module) |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 3D memory usage (64³ = 262k vs 8k for 64×128) | Start with smaller grids (32³), optimize later |
| Performance regression | JAX JIT handles 3D arrays well; profile early |
| Validation coverage | Two analytic test cases with convergence checks |
| Breaking existing workflows | Full replacement means clean break; document migration |
