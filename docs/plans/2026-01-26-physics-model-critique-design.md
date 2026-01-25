# Physics Model Critique and Improvement Roadmap

**Date:** 2026-01-26
**Status:** Analysis Complete
**Scope:** Critical review of JAX-FRC physics models and solvers for FRC fusion applications

## Executive Summary

This document provides a systematic critique of the three physics models (Resistive MHD, Extended MHD, Hybrid Kinetic) and supporting infrastructure in JAX-FRC. The analysis identifies correctness issues, missing physics, and performance limitations, with prioritized recommendations for improvement.

**Key findings:**
- Hybrid Kinetic model has a missing Hall term in the E-field calculation (critical bug)
- Extended MHD cylindrical curl operators have incorrect (1/r) factors
- No collision operator in Hybrid limits runs to sub-collision timescales
- Semi-implicit solver is over-simplified (uniform damping vs k-dependent)
- No ∇·B divergence cleaning in field evolution

---

## 1. Resistive MHD Analysis

### 1.1 Strengths

| Feature | Assessment |
|---------|------------|
| Flux function (ψ) formulation | Correct for 2D axisymmetric geometry |
| Chodura anomalous resistivity | Appropriate for θ-pinch formation reconnection |
| Δ* operator | Correctly includes -(1/r)∂ψ/∂r term |
| Diffusion CFL | Properly computed from η_max |

### 1.2 Limitations

#### 1.2.1 No Stability Prediction (Fundamental)

Single-fluid MHD lacks the Hall effect and FLR physics that stabilize real FRCs against the n=1 tilt mode. This is documented but worth emphasizing: **Resistive MHD should only be used for formation dynamics and circuit coupling, never for stability assessment.**

#### 1.2.2 No External Coil Representation

Current implementation uses hardcoded Dirichlet boundaries (ψ=0). Real FRC experiments require:

```
ψ_boundary = Σ_i G(r,z; r_i,z_i) * I_i
```

Where G is the Green's function for a current loop. This enables:
- Flux control during formation
- Wall stabilization feedback
- Realistic field null geometry

#### 1.2.3 No Toroidal Field Evolution

The model forces B_φ = 0. FRCs can develop weak toroidal fields from:
- n=2 rotational instability
- Field-reversed θ-pinch formation asymmetries
- Neutral beam injection (future capability)

#### 1.2.4 Boundary Condition Artifacts

The use of `jnp.roll` for finite differences creates implicit periodic conditions before explicit BCs are applied. For steep gradients near walls, this introduces numerical artifacts in the first/last grid points.

**Recommendation:** Use one-sided differences at boundaries or ghost cells.

---

## 2. Extended MHD Analysis

### 2.1 Strengths

| Feature | Assessment |
|---------|------------|
| Hall term (J×B)/(ne) | Key physics for FRC tilt stabilization |
| Electron pressure gradient | Correct two-fluid Ohm's law |
| Halo density model | Avoids 1/n singularities in vacuum |
| Semi-implicit framework | Addresses Whistler stiffness |

### 2.2 Limitations

#### 2.2.1 Semi-Implicit Solver Over-Simplified (Critical)

**Current implementation:**
```python
implicit_factor = 1.0 / (1.0 + dt**2 * self.damping_factor)
dB_implicit = dB_explicit * implicit_factor
```

This applies uniform damping to all modes. The correct semi-implicit scheme (NIMROD, HiFi) solves:

```
(I - Δt² L_Hall) ΔB^{n+1} = Explicit_RHS
```

Where L_Hall has k-dependent eigenvalues. Uniform damping:
- Over-damps large-scale (low-k) dynamics of interest
- Under-damps high-k modes that cause instability

**Recommendation:** Implement spectral or multigrid solve for the implicit operator.

#### 2.2.2 Missing Gyroviscosity

Extended MHD codes for FRC typically include the gyroviscous stress tensor:

```
π_gv = (p_⊥ / (4 Ω_ci)) [b × (∇v + (∇v)^T) + ...]
```

This captures finite-Larmor-radius (FLR) stabilization in fluid form. The Braginskii or CGL closure is standard.

#### 2.2.3 No Energy Equation

The model evolves B but not temperature. FRC confinement requires:

```
(3/2) n dT/dt = -∇·q - p∇·v + η J² + Q_aux
```

Where:
- q = heat flux (parallel + perpendicular)
- η J² = Ohmic heating
- Q_aux = auxiliary heating (NBI, compression)

#### 2.2.4 Incorrect Cylindrical Curl (Bug)

In `_compute_current` (lines 111-126), J_z is computed as:

```python
J_z = (1.0 / MU0) * dB_phi_dr
```

The correct cylindrical form is:

```python
J_z = (1.0 / MU0) * (1/r) * d(r * B_phi)/dr
     = (1.0 / MU0) * (B_phi/r + dB_phi_dr)
```

This introduces O(1) errors near the magnetic axis.

**Recommendation:** Fix curl operators throughout. Consider using a staggered (Yee) grid.

#### 2.2.5 Static Density

The halo model applies a fixed radial profile. Real FRCs have dynamic density from:
- Cross-field particle transport
- End losses along open field lines
- Ionization of edge neutrals
- Wall recycling

---

## 3. Hybrid Kinetic Analysis

### 3.1 Strengths

| Feature | Assessment |
|---------|------------|
| Delta-f PIC method | Reduces noise by O(1/δf) vs full-f |
| Boris pusher | Standard, energy-conserving algorithm |
| Rigid rotor equilibrium f₀ | Appropriate for rotating FRCs |
| Weight evolution | Correctly derived from d(ln f₀)/dt |
| CIC deposition | Standard second-order shape function |

### 3.2 Limitations

#### 3.2.1 Missing Hall Term in E-field (Critical Bug)

**Current implementation** (`compute_rhs`, lines 84-98):

```python
E_r = -dp_dr / ne + self.eta * J_total[:, :, 0]
E_phi = self.eta * J_total[:, :, 1]
E_z = -dp_dz / ne + self.eta * J_total[:, :, 2]
```

This computes E = -∇p_e/(ne) + ηJ, but the correct hybrid Ohm's law is:

```
E = -v_e × B + ηJ - ∇p_e/(ne)
  = -(v_i - J/(ne)) × B + ηJ - ∇p_e/(ne)
  = -v_i × B + (J × B)/(ne) + ηJ - ∇p_e/(ne)
```

The Hall term (J × B)/(ne) is missing. This fundamentally changes the physics.

**Recommendation:** Add Hall term immediately. This is a ~20-line fix with major physics impact.

#### 3.2.2 No Collision Operator (Critical)

Delta-f methods require collisions to:
1. Prevent unbounded growth of δf
2. Provide physical thermalization
3. Enable steady-state solutions

Without collisions, particle weights grow secularly and the simulation becomes unphysical after a few collision times (τ_ii ~ 10-100 μs for FRC parameters).

**Minimum viable:** Krook operator with `dw/dt += -ν * w`

**Proper implementation:** Fokker-Planck collision operator (Takizuka-Abe or Langevin)

#### 3.2.3 Reflecting Particle Boundaries (Incorrect Physics)

Current implementation reflects all particles at domain boundaries. Real FRCs have:

| Region | Boundary Condition |
|--------|-------------------|
| Closed field lines (core) | Particles confined, follow flux surfaces |
| Open field lines (SOL) | Particles escape along field to end walls |
| Separatrix | Determines which regime applies |

**Recommendation:** Implement field-line tracing to determine if particle is on open/closed field line, then apply appropriate BC.

#### 3.2.4 No Particle Sub-cycling

The `HybridSolver` uses the same dt for fields and particles:

```python
def step(self, state, dt, model, geometry):
    rhs = model.compute_rhs(state, geometry)  # Field RHS
    new_B = state.B + dt * rhs.B              # Field update
    state = model.push_particles(state, geometry, dt)  # Particle push
```

Particles require dt < 0.1/Ω_ci (cyclotron), while fields can use dt ~ dx/v_A (Alfvén). For typical FRC parameters:
- Ω_ci ~ 10⁷ rad/s → dt_particle ~ 10 ns
- v_A ~ 10⁶ m/s, dx ~ 1 cm → dt_field ~ 10 ns to 100 ns

Sub-cycling particles (multiple particle steps per field step) can provide 10-100× speedup.

#### 3.2.5 Quasi-neutrality Implementation

Setting `J_total = J_i` (line 82) assumes instantaneous electron response. The correct hybrid closure is:

```
n_e = n_i  (quasi-neutrality)
v_e = v_i - J/(n_e e)  (from J = ne(v_i - v_e))
```

The electron convection contribution to the E-field is lost in the current formulation.

---

## 4. Equilibrium Solver Analysis

### 4.1 Strengths

| Feature | Assessment |
|---------|------------|
| Grad-Shafranov formulation | Correct for axisymmetric equilibrium |
| SOR iteration | Valid classical method |
| Parameterized profiles | Enables p(ψ), F(ψ) specification |

### 4.2 Limitations

#### 4.2.1 SOR Performance

For a 128×256 grid, SOR requires O(10⁴-10⁵) iterations. Alternatives:

| Method | Iterations | Speedup |
|--------|------------|---------|
| SOR (current) | 10⁴-10⁵ | 1× |
| Multigrid | 10-50 | 100-1000× |
| Newton-Krylov | 5-20 (outer) | 100-500× |

JAX's `jax.scipy.sparse.linalg.cg` or `gmres` could accelerate significantly.

#### 4.2.2 Fixed-Boundary Only

The solver assumes ψ is specified at boundaries. Free-boundary FRC equilibria require:
- Green's function coupling to external coils
- Iterative separatrix location update
- Vacuum field matching

#### 4.2.3 No Equilibrium Figures of Merit

The code doesn't compute standard FRC metrics:

| Quantity | Definition | Typical FRC Value |
|----------|------------|-------------------|
| ⟨β⟩ | 2μ₀⟨p⟩/⟨B²⟩ | 0.8-0.95 |
| r_s/r_w | Separatrix/wall radius ratio | 0.4-0.7 |
| κ = L_s/(2r_s) | Elongation | 3-10 |
| x_s = r_s/r_Δ | Separatrix parameter | 0.3-0.5 |

---

## 5. Cross-Cutting Numerical Issues

### 5.1 No Divergence Cleaning

Extended MHD and Hybrid evolve B via Faraday's law. Numerical errors accumulate ∇·B ≠ 0, causing:
- Unphysical parallel forces on particles
- Energy conservation violations
- Numerical instability

**Solutions:**
1. Projection method: B → B - ∇φ where ∇²φ = ∇·B
2. Hyperbolic cleaning: ∂ψ/∂t + c²_h ∇·B = -ψ/τ
3. Constrained transport (staggered grid)

### 5.2 Axis Singularity (r = 0)

All 1/r terms become singular at the axis. Current code doesn't special-case this. Options:
- L'Hôpital's rule: lim(r→0) (1/r)∂f/∂r = ∂²f/∂r²
- Shifted grid: Place r grid points at r = dr/2, 3dr/2, ...
- Regularity conditions: Enforce f(r=0) = 0 or ∂f/∂r(r=0) = 0

### 5.3 Second-Order Accuracy Throughout

All spatial derivatives use central differences (2nd order). For FRC with steep gradients at separatrix, higher-order methods (4th order compact, spectral) would improve accuracy per grid point.

---

## 6. Prioritized Recommendations

### Tier 1: Correctness (Must Fix)

| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 1 | Add Hall term to Hybrid E-field | `hybrid_kinetic.py:84-98` | Low | Critical |
| 2 | Fix cylindrical curl (1/r) factors | `extended_mhd.py:111-126` | Low | High |
| 3 | Add collision operator to Hybrid | `hybrid_kinetic.py` | Medium | Critical |
| 4 | Add ∇·B divergence cleaning | All field evolution | Medium | High |

### Tier 2: Physics Fidelity

| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 5 | Open field line particle losses | `hybrid_kinetic.py:230-252` | Medium | High |
| 6 | Energy equation in Extended MHD | `extended_mhd.py` | High | High |
| 7 | Proper semi-implicit Hall operator | `semi_implicit.py` | High | Medium |
| 8 | Gyroviscosity in Extended MHD | `extended_mhd.py` | High | Medium |

### Tier 3: Performance

| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 9 | Particle sub-cycling | `semi_implicit.py:101-128` | Medium | High |
| 10 | Multigrid GS solver | `grad_shafranov.py` | High | High |
| 11 | Spectral methods for semi-implicit | `semi_implicit.py` | High | Medium |

### Tier 4: Full Capability

| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 12 | Free-boundary equilibrium | `grad_shafranov.py` | Very High | High |
| 13 | External coil coupling | `resistive_mhd.py` | High | Medium |
| 14 | Neutral beam injection | New module | Very High | High |

---

## 7. Recommended Implementation Order

**Phase 1: Correctness (Before trusting any results)**
1. Fix Hall term in Hybrid E-field (~1 day)
2. Fix cylindrical curl operators (~1 day)
3. Add basic Krook collision operator (~2-3 days)
4. Add projection-based ∇·B cleaning (~2-3 days)

**Phase 2: Validation**
- Re-run existing test cases after Phase 1 fixes
- Compare with published FRC benchmarks (e.g., Belova cases)
- Verify energy/particle conservation

**Phase 3: Physics Extensions**
- Energy equation and transport
- Open/closed field line particle BCs
- Gyroviscosity

**Phase 4: Performance**
- Particle sub-cycling
- Multigrid equilibrium solver
- GPU optimization of particle operations

---

## 8. References

1. Belova, E.V. et al. "Numerical study of tilt stability of prolate field-reversed configurations" Phys. Plasmas 7, 4996 (2000)
2. Sovinec, C.R. et al. "Nonlinear magnetohydrodynamics simulation using high-order finite elements" J. Comput. Phys. 195, 355 (2004) [NIMROD]
3. Brackbill, J.U. "FLIP MHD: A particle-in-cell method for magnetohydrodynamics" J. Comput. Phys. 96, 163 (1991)
4. Parker, S.E. and Lee, W.W. "A fully nonlinear characteristic method for gyrokinetic simulation" Phys. Fluids B 5, 77 (1993) [Delta-f]

---

## Appendix: Code Snippets for Priority Fixes

### A.1 Hall Term Fix (Priority 1)

```python
# In hybrid_kinetic.py compute_rhs(), after line 93:

# Compute Hall term: (J × B) / (ne)
J_r, J_phi, J_z = J_total[:,:,0], J_total[:,:,1], J_total[:,:,2]
B_r, B_phi, B_z = state.B[:,:,0], state.B[:,:,1], state.B[:,:,2]

hall_r = (J_phi * B_z - J_z * B_phi) / ne
hall_phi = (J_z * B_r - J_r * B_z) / ne
hall_z = (J_r * B_phi - J_phi * B_r) / ne

E_r = -dp_dr / ne + self.eta * J_r + hall_r
E_phi = self.eta * J_phi + hall_phi
E_z = -dp_dz / ne + self.eta * J_z + hall_z
```

### A.2 Cylindrical Curl Fix (Priority 2)

```python
# In extended_mhd.py _compute_current():

# J_z = (1/mu_0) * (1/r) * d(r*B_phi)/dr
r = geometry.r_grid  # Need to pass geometry
rB_phi = r * B_phi
d_rB_phi_dr = (jnp.roll(rB_phi, -1, axis=0) - jnp.roll(rB_phi, 1, axis=0)) / (2 * dr)
J_z = (1.0 / MU0) * d_rB_phi_dr / r
```

### A.3 Krook Collision Operator (Priority 3)

```python
# In hybrid_kinetic.py, add to push_particles():

# Krook collision operator: dw/dt = -nu * w
nu_ii = self.collision_frequency  # New parameter, ~10^4 s^-1 for FRC
w_new = w_new * jnp.exp(-nu_ii * dt)
```
