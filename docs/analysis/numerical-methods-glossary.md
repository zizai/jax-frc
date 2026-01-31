# Numerical Methods Glossary and Technical Guide

## Part 1: Abbreviations and Technical Terms

### General Numerical Methods

| Term | Full Name | Description |
|------|-----------|-------------|
| **FVM** | Finite Volume Method | Discretization that integrates equations over control volumes; naturally conservative |
| **FDM** | Finite Difference Method | Discretization using derivative approximations at grid points |
| **FEM** | Finite Element Method | Discretization using basis functions over elements; common in engineering |
| **DG** | Discontinuous Galerkin | Hybrid of FVM and FEM; polynomial solutions within cells, discontinuous at interfaces |
| **DOF** | Degrees of Freedom | Number of independent variables in the discretization |

### Time Integration

| Term | Full Name | Description |
|------|-----------|-------------|
| **RK** | Runge-Kutta | Family of explicit multi-stage time integrators |
| **RK2/RK3/RK4** | Runge-Kutta 2nd/3rd/4th order | Specific RK methods with corresponding accuracy |
| **SSP** | Strong Stability Preserving | RK methods that preserve TVD property of spatial discretization |
| **SSPRK(s,p)** | SSP Runge-Kutta | s stages, p-th order accurate |
| **IMEX** | Implicit-Explicit | Treats stiff terms implicitly, non-stiff explicitly |
| **STS** | Super Time Stepping | Accelerated explicit method for parabolic (diffusion) terms |
| **RKL** | Runge-Kutta Legendre | STS variant using Legendre polynomials for stability |
| **VL2** | Van Leer 2nd order | Predictor-corrector scheme, similar to MUSCL-Hancock |
| **CFL** | Courant-Friedrichs-Lewy | Stability condition: dt ≤ CFL × dx / max_wavespeed |

### Spatial Reconstruction

| Term | Full Name | Description |
|------|-----------|-------------|
| **PLM** | Piecewise Linear Method | 2nd-order reconstruction with slope limiters |
| **PPM** | Piecewise Parabolic Method | 3rd-order reconstruction using parabolas |
| **MUSCL** | Monotonic Upstream-centered Scheme for Conservation Laws | Framework for high-order FVM reconstruction |
| **TVD** | Total Variation Diminishing | Property ensuring no new extrema are created |
| **WENO** | Weighted Essentially Non-Oscillatory | High-order reconstruction avoiding oscillations near discontinuities |
| **MP5** | Monotonicity Preserving 5th order | 5th-order scheme that preserves monotonicity |
| **MC** | Monotonized Central | Popular slope limiter (also called van Leer limiter) |

### Riemann Solvers

| Term | Full Name | Description |
|------|-----------|-------------|
| **HLL** | Harten-Lax-van Leer | 2-wave approximate Riemann solver; robust but diffusive |
| **HLLC** | HLL-Contact | HLL with contact wave; 3-wave solver for hydrodynamics |
| **HLLD** | HLL-Discontinuities | HLL with all discontinuities; 7-wave solver for MHD |
| **LHLLD** | Low-dissipation HLLD | HLLD variant with reduced numerical diffusion |
| **LLF** | Local Lax-Friedrichs | Simple, very diffusive Riemann solver |
| **Roe** | Roe's linearized solver | Exact linearized Riemann solver; most accurate but expensive |

### Divergence Cleaning

| Term | Full Name | Description |
|------|-----------|-------------|
| **CT** | Constrained Transport | Staggered grid method maintaining ∇·B = 0 exactly |
| **GLM** | Generalized Lagrange Multiplier | Hyperbolic divergence cleaning method |
| **EMF** | Electromotive Force | E-field at cell edges in CT; E = -v × B |
| **UCT** | Upwind Constrained Transport | CT with upwind EMF averaging |

### Physics Models

| Term | Full Name | Description |
|------|-----------|-------------|
| **MHD** | Magnetohydrodynamics | Fluid description of electrically conducting plasma |
| **HD** | Hydrodynamics | Fluid dynamics without magnetic fields |
| **RMHD** | Relativistic MHD | MHD with special relativistic effects |
| **GRMHD** | General Relativistic MHD | MHD in curved spacetime |
| **CGL** | Chew-Goldberger-Low | Anisotropic pressure MHD model |
| **EOS** | Equation of State | Relation between pressure, density, and temperature |

### FRC-Specific Terms

| Term | Full Name | Description |
|------|-----------|-------------|
| **FRC** | Field-Reversed Configuration | Compact toroid with closed field lines, no toroidal field |
| **Separatrix** | - | Boundary between open and closed field lines |
| **O-point** | - | Magnetic null where field lines close on themselves |
| **X-point** | - | Magnetic null where field lines cross (reconnection site) |
| **η** | Resistivity | Magnetic diffusion coefficient (Ohm's law: E = ηJ) |
| **δi** | Ion inertial length | Scale where Hall physics becomes important: δi = c/ωpi |

---

## Part 2: Why Divergence Cleaning Matters

### The Problem: ∇·B = 0

Maxwell's equations require that magnetic field has no monopoles:

```
∇·B = 0    (always, everywhere)
```

This is a **constraint**, not an evolution equation. The MHD equations are:

```
∂ρ/∂t + ∇·(ρv) = 0                           (mass)
∂(ρv)/∂t + ∇·(ρvv + P - BB/μ₀) = 0           (momentum)
∂E/∂t + ∇·((E + P)v - B(v·B)/μ₀) = 0         (energy)
∂B/∂t + ∇×E = 0,  where E = -v×B             (induction)
```

**The induction equation preserves ∇·B = 0 analytically:**

```
∂(∇·B)/∂t = ∇·(∂B/∂t) = ∇·(-∇×E) = 0
```

If ∇·B = 0 initially, it stays zero forever... **in continuous mathematics**.

### Why Numerical Methods Break This

Discrete approximations introduce errors:

1. **Truncation error**: Finite differences approximate derivatives
2. **Round-off error**: Floating-point arithmetic is inexact
3. **Non-conservative fluxes**: Some schemes don't conserve flux exactly

These errors create **magnetic monopoles** (∇·B ≠ 0) that:

- Cause unphysical forces parallel to B
- Corrupt magnetic topology (spurious reconnection)
- Accumulate over time (errors grow)
- Can cause numerical instability

### Consequences for FRC Simulation

In FRC, magnetic topology is critical:

```
                    Separatrix
                   /
    =============/================  Open field lines
                |     O-point     |
    -----------|       ⊙        |-------  Closed field lines
                |                 |
    ============\================  Open field lines
                   \
                    X-point (reconnection)
```

Divergence errors can:
- Move the separatrix incorrectly
- Create spurious reconnection
- Change the O-point location
- Corrupt flux conservation

---

## Part 3: Divergence Cleaning Methods

### Method 1: Projection (Hodge Decomposition)

**Idea:** After each time step, project B onto divergence-free space.

**Algorithm:**
```
1. Compute divergence error: φ = ∇·B
2. Solve Poisson equation: ∇²ψ = φ
3. Correct field: B_clean = B - ∇ψ
```

**Why it works:**
- Any vector field can be decomposed: B = B_div-free + ∇ψ
- Subtracting ∇ψ removes the curl-free (divergent) part
- Result is exactly divergence-free

**Implementation:**
```python
def projection_clean(B, geometry):
    """Project B onto divergence-free space."""
    # Step 1: Compute divergence
    div_B = divergence(B, geometry)

    # Step 2: Solve Poisson equation ∇²ψ = ∇·B
    psi = poisson_solve(div_B, geometry)

    # Step 3: Correct B
    grad_psi = gradient(psi, geometry)
    B_clean = B - grad_psi

    return B_clean
```

**Pros:**
- Simple to understand
- Works on any grid

**Cons:**
- Requires global Poisson solve (expensive, non-local)
- Not conservative (changes total magnetic energy)
- Must be applied every time step

---

### Method 2: Hyperbolic Divergence Cleaning (GLM/Dedner)

**Idea:** Add a scalar field ψ that transports divergence errors away and damps them.

**Modified equations:**
```
∂B/∂t + ∇×E + ∇ψ = 0
∂ψ/∂t + c_h²∇·B = -c_p²ψ/c_h
```

Where:
- **c_h**: Hyperbolic wave speed (how fast errors propagate)
- **c_p**: Parabolic damping rate (how fast errors decay)

**Why it works:**
- Divergence errors become waves that propagate at speed c_h
- Waves exit through boundaries or get damped
- System is hyperbolic, fits naturally into Godunov framework

**Implementation:**
```python
@dataclass
class GLMState:
    B: Array      # Magnetic field (3 components)
    psi: Array    # Divergence cleaning scalar

def glm_flux(state_L, state_R, normal, c_h):
    """Compute GLM-modified flux at interface."""
    # Standard MHD flux
    F_mhd = hlld_flux(state_L, state_R, normal)

    # GLM modification to B-normal flux
    B_n_L = dot(state_L.B, normal)
    B_n_R = dot(state_R.B, normal)
    psi_L = state_L.psi
    psi_R = state_R.psi

    # Upwind flux for B_n and psi (exact Riemann solution)
    F_Bn = 0.5 * (psi_L + psi_R) - 0.5 * c_h * (B_n_R - B_n_L)
    F_psi = 0.5 * c_h**2 * (B_n_L + B_n_R) - 0.5 * c_h * (psi_R - psi_L)

    return F_mhd, F_Bn, F_psi

def glm_source(state, c_h, c_p, dt):
    """Apply GLM damping source term."""
    # Exponential decay of psi
    decay = exp(-c_p**2 / c_h * dt)
    return state.replace(psi=state.psi * decay)
```

**Parameter choices:**
```python
# Typical values
c_h = max_wavespeed  # Usually fastest MHD wave
c_p = 0.18 * c_h     # Dedner's recommendation
# Or use α = c_p²/(c_h * c_r) where c_r = min(dx, dy, dz)
alpha = 0.4  # Typical value
```

**Pros:**
- Local operations only (no global solve)
- Fits naturally into finite volume framework
- Hyperbolic, so explicit time stepping works

**Cons:**
- Only reduces errors, doesn't eliminate them
- Adds computational cost (extra variable ψ)
- Requires tuning of c_h and c_p

**Reference:** Dedner et al., JCP 175 (2002)

---

### Method 3: Powell 8-Wave Formulation

**Idea:** Add source terms proportional to ∇·B to cancel unphysical effects.

**Modified equations:**
```
∂(ρv)/∂t + ... = -(∇·B)B/μ₀
∂E/∂t + ... = -(∇·B)(v·B)/μ₀
∂B/∂t + ... = -(∇·B)v
```

**Why it works:**
- Source terms cancel the unphysical parallel force from monopoles
- Treats ∇·B as an 8th wave in the MHD system

**Pros:**
- Simple to implement
- No extra variables or equations

**Cons:**
- **Non-conservative** (violates conservation laws)
- Errors still accumulate
- Not recommended for production simulations

**Reference:** Powell et al., JCP 154 (1999)

---

## Part 4: Constrained Transport (CT)

### The Key Insight

Instead of cleaning divergence errors after they occur, **prevent them from forming**.

**Stokes' theorem:**
```
∮ B·dA = ∫∫ (∇·B) dV
```

If we ensure the flux through every cell face is consistent, ∇·B = 0 automatically.

### Staggered Grid Layout

CT uses a **staggered grid** where different quantities live at different locations:

```
        Bz (face)
           |
    -------|-------
    |      |      |
    |   ρ,p,E    Bx (face)
    |   (cell)   |
    -------|-------
           |
        Ez (edge)
```

**Grid locations:**
- **Cell centers**: ρ, p, E, v (scalar and vector quantities)
- **Face centers**: B components (normal to face)
- **Edge centers**: E components (along edge)

```
     +-------+-------+
     |       |       |
     |  Bz   |  Bz   |   Bz at z-faces
     |       |       |
     +--Ex---+--Ex---+   Ex at x-edges
     |       |       |
Bx   |   •   |   •   |   Cell centers (ρ, p, v)
     |       |       |
     +--Ex---+--Ex---+
     |       |       |
     |  Bz   |  Bz   |
     |       |       |
     +-------+-------+
         By at y-faces
```

### The CT Update Algorithm

**Faraday's law in integral form:**
```
d/dt ∫∫ B·dA = -∮ E·dl
```

For a face with area A and boundary edges:
```
dB_face/dt = -(1/A) × (sum of E along edges)
```

**2D Example (Bz update):**
```
     Ex(j+1)
    ←-------
    |       |
Ey(i)|  Bz  |Ey(i+1)
    |       |
    -------→
     Ex(j)

dBz/dt = -(1/ΔxΔy) × [Ex(j+1)Δx - Ex(j)Δx + Ey(i+1)Δy - Ey(i)Δy]
       = -[(Ex(j+1) - Ex(j))/Δy - (Ey(i+1) - Ey(i))/Δx]
       = -(∇×E)_z
```

This is **exactly** the curl, so ∇·B = 0 is preserved to machine precision!

### Computing the EMF (Electric Field)

The challenge: E lives at edges, but Riemann solvers give fluxes at faces.

**Step 1: Riemann solver at faces**
```python
def face_flux(state_L, state_R, direction):
    """Compute MHD flux at face center."""
    # HLLD or other Riemann solver
    flux = hlld_riemann(state_L, state_R, direction)
    # Extract E = -v×B from flux
    # For x-face: Ey = -vx*By + vy*Bx, Ez = -vx*Bz + vz*Bx
    return flux, E_face
```

**Step 2: Average to edges (EMF reconstruction)**

Multiple methods exist:

**Simple arithmetic average:**
```python
def emf_arithmetic(E_faces):
    """Average face E-fields to edges."""
    # Ez at (i+1/2, j+1/2) from 4 neighboring faces
    Ez_edge = 0.25 * (Ez_xface[i,j] + Ez_xface[i,j+1] +
                      Ez_yface[i,j] + Ez_yface[i+1,j])
    return Ez_edge
```

**Upwind CT (UCT-HLL):**
```python
def emf_uct_hll(state, E_faces, geometry):
    """Upwind EMF averaging using HLL wavespeeds."""
    # Get wavespeeds at corners
    S_L, S_R = hll_wavespeeds(state, geometry)

    # Upwind average based on wave direction
    if S_L > 0:
        Ez_edge = Ez_from_left
    elif S_R < 0:
        Ez_edge = Ez_from_right
    else:
        # Weighted average
        Ez_edge = (S_R * Ez_L - S_L * Ez_R + S_L * S_R * (Bx_R - Bx_L)) / (S_R - S_L)

    return Ez_edge
```

### Full CT Implementation

```python
@dataclass
class CTState:
    """State with staggered magnetic field."""
    rho: Array      # Cell-centered density
    momentum: Array # Cell-centered momentum
    energy: Array   # Cell-centered energy
    Bx: Array       # x-face-centered Bx
    By: Array       # y-face-centered By
    Bz: Array       # z-face-centered Bz

def ct_step(state: CTState, dt: float, geometry: Geometry) -> CTState:
    """Single CT time step."""

    # Step 1: Interpolate B to cell centers for Riemann solver
    B_cc = interpolate_to_centers(state.Bx, state.By, state.Bz)

    # Step 2: Reconstruct interface states (PLM/PPM)
    states_L, states_R = reconstruct(state, geometry)

    # Step 3: Solve Riemann problems at faces, get fluxes and E-fields
    flux_x, Ex_xface, Ey_xface, Ez_xface = riemann_x(states_L, states_R)
    flux_y, Ex_yface, Ey_yface, Ez_yface = riemann_y(states_L, states_R)
    flux_z, Ex_zface, Ey_zface, Ez_zface = riemann_z(states_L, states_R)

    # Step 4: Average E-fields to edges (EMF reconstruction)
    Ex_edge = emf_average_x(Ey_zface, Ez_yface)  # Ex at y-z edges
    Ey_edge = emf_average_y(Ex_zface, Ez_xface)  # Ey at x-z edges
    Ez_edge = emf_average_z(Ex_yface, Ey_xface)  # Ez at x-y edges

    # Step 5: Update cell-centered quantities (standard FVM)
    new_rho = state.rho - dt * divergence(flux_rho, geometry)
    new_mom = state.momentum - dt * divergence(flux_mom, geometry)
    new_E = state.energy - dt * divergence(flux_E, geometry)

    # Step 6: Update face-centered B using CT (Faraday's law)
    # dBx/dt = -(∂Ez/∂y - ∂Ey/∂z)
    new_Bx = state.Bx - dt * (diff_y(Ez_edge) - diff_z(Ey_edge))
    # dBy/dt = -(∂Ex/∂z - ∂Ez/∂x)
    new_By = state.By - dt * (diff_z(Ex_edge) - diff_x(Ez_edge))
    # dBz/dt = -(∂Ey/∂x - ∂Ex/∂y)
    new_Bz = state.Bz - dt * (diff_x(Ey_edge) - diff_y(Ex_edge))

    return CTState(new_rho, new_mom, new_E, new_Bx, new_By, new_Bz)

def diff_x(E_edge):
    """Compute ∂E/∂x at face from edge values."""
    return (E_edge[1:,:,:] - E_edge[:-1,:,:]) / dx

def diff_y(E_edge):
    """Compute ∂E/∂y at face from edge values."""
    return (E_edge[:,1:,:] - E_edge[:,:-1,:]) / dy

def diff_z(E_edge):
    """Compute ∂E/∂z at face from edge values."""
    return (E_edge[:,:,1:] - E_edge[:,:,:-1]) / dz
```

### Why CT Preserves ∇·B = 0

**Discrete divergence:**
```
(∇·B)_cell = (Bx[i+1] - Bx[i])/Δx + (By[j+1] - By[j])/Δy + (Bz[k+1] - Bz[k])/Δz
```

**Time derivative of divergence:**
```
d(∇·B)/dt = ∇·(dB/dt)
          = ∇·(-∇×E)
          = 0  (divergence of curl is zero)
```

This holds **exactly** in the discrete case because:
- Each edge E appears in exactly two face updates with opposite signs
- The contributions cancel perfectly in the divergence sum

### CT in Curvilinear Coordinates

For cylindrical (r, φ, z) coordinates, the CT update becomes:

```python
def ct_cylindrical(state, E_edge, geometry):
    """CT update in cylindrical coordinates."""
    r = geometry.r
    dr, dphi, dz = geometry.dr, geometry.dphi, geometry.dz

    # Br update: dBr/dt = -(1/r)(∂Ez/∂φ) + ∂Eφ/∂z
    dBr = -dt * ((1/r) * diff_phi(Ez_edge) - diff_z(Ephi_edge))

    # Bφ update: dBφ/dt = ∂Er/∂z - ∂Ez/∂r
    dBphi = -dt * (diff_z(Er_edge) - diff_r(Ez_edge))

    # Bz update: dBz/dt = -(1/r)(∂(r*Eφ)/∂r - ∂Er/∂φ)
    dBz = -dt * ((1/r) * diff_r(r * Ephi_edge) - (1/r) * diff_phi(Er_edge))

    return state.Bx + dBr, state.By + dBphi, state.Bz + dBz
```

---

## Part 5: Comparison Summary

| Method | ∇·B Error | Cost | Complexity | Conservation |
|--------|-----------|------|------------|--------------|
| **None** | Grows unbounded | - | - | Yes |
| **Projection** | Zero (after clean) | High (Poisson) | Low | No |
| **GLM/Dedner** | Small, bounded | Low | Medium | Yes |
| **Powell 8-wave** | Bounded | Low | Low | **No** |
| **Constrained Transport** | **Machine zero** | Medium | High | Yes |

### Recommendation for FRC

**For production FRC simulations: Use Constrained Transport**

Reasons:
1. FRC topology is sensitive to divergence errors
2. Long-time simulations accumulate errors
3. CT maintains ∇·B = 0 exactly
4. Well-established in MHD community

**For prototyping/testing: GLM is acceptable**

Reasons:
1. Simpler to implement
2. Works on any grid (no staggering)
3. Errors bounded, not catastrophic

---

## References

1. **Evans & Hawley** (1988), "Simulation of Magnetohydrodynamic Flows: A Constrained Transport Method", ApJ 332, 659
2. **Dedner et al.** (2002), "Hyperbolic Divergence Cleaning for the MHD Equations", JCP 175, 645
3. **Tóth** (2000), "The ∇·B = 0 Constraint in Shock-Capturing Magnetohydrodynamics Codes", JCP 161, 605
4. **Gardiner & Stone** (2005), "An unsplit Godunov method for ideal MHD via constrained transport", JCP 205, 509
5. **Mignone & Tzeferacos** (2010), "A second-order unsplit Godunov scheme for cell-centered MHD", JCP 229, 2117
