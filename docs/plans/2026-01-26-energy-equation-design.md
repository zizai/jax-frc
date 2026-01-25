# Energy Equation Implementation Design

**Date:** 2026-01-26
**Status:** Approved
**Scope:** Add temperature evolution with anisotropic thermal transport to Extended MHD

## Executive Summary

This design adds temperature evolution to the Extended MHD model, enabling simulation of thermal transport, Ohmic heating, and compressional heating in FRCs. The implementation uses full anisotropic thermal conduction with separate parallel and perpendicular conductivities.

**Key deliverables:**
- Temperature field in State dataclass
- ThermalTransport class for anisotropic conduction
- Temperature RHS computation in ExtendedMHD
- Semi-implicit treatment of stiff conduction term

---

## 1. Physics Background

### 1.1 The Energy Equation

From plasma physics, the temperature evolution equation is:

```
(3/2) n ∂T/∂t + (3/2) n v·∇T = -p∇·v - ∇·q + ηJ² + Q_aux
```

Where:
- `-p∇·v`: Compressional heating/cooling (PdV work)
- `-∇·q`: Heat flux divergence (thermal conduction)
- `ηJ²`: Ohmic/resistive heating
- `Q_aux`: Auxiliary heating sources (NBI, compression)

### 1.2 Anisotropic Heat Flux

The heat flux in magnetized plasma is strongly anisotropic:

```
q = -κ_∥ (b·∇T) b - κ_⊥ (I - bb)·∇T
```

Where:
- `b = B/|B|` is the unit vector along the magnetic field
- `κ_∥` is the parallel thermal conductivity
- `κ_⊥` is the perpendicular thermal conductivity

For FRCs, `κ_∥/κ_⊥ ~ 10⁶`, meaning heat flows primarily along field lines.

### 1.3 Spitzer Conductivity

The parallel conductivity follows Spitzer's formula:

```
κ_∥ = κ_0 T^(5/2) / Z ln(Λ)
```

Where:
- `κ_0 ≈ 3.16 × 10²⁰` [W/(m·eV^(7/2))] for electrons
- `Z` is the ion charge
- `ln(Λ)` is the Coulomb logarithm (~15-20)

---

## 2. Implementation Design

### 2.1 State Modifications

Extend the `State` dataclass to include temperature:

```python
@dataclass(frozen=True)
class State:
    # Existing scalar fields (nr, nz)
    psi: Array        # Poloidal flux function
    n: Array          # Number density
    p: Array          # Pressure (now derived: p = n * T)

    # NEW: Temperature field
    T: Array          # Temperature (nr, nz) [eV]

    # Vector fields (nr, nz, 3)
    B: Array          # Magnetic field
    E: Array          # Electric field
    v: Array          # Fluid velocity

    # ...rest unchanged...
```

Update `State.zeros()` and pytree registration to include T.

### 2.2 ThermalTransport Class

New file: `jax_frc/models/energy.py`

```python
@dataclass
class ThermalTransport:
    """Anisotropic thermal transport model for magnetized plasma.

    Computes heat flux: q = -κ_∥ (b·∇T) b - κ_⊥ (∇T - (b·∇T) b)

    Attributes:
        kappa_parallel_0: Base parallel conductivity [W/(m·eV)]
        kappa_perp_ratio: Ratio κ_⊥/κ_∥ (typically 1e-6)
        use_spitzer: If True, κ_∥ = κ_0 T^(5/2)
        coulomb_log: Coulomb logarithm (default 15)
    """
    kappa_parallel_0: float = 3.16e20
    kappa_perp_ratio: float = 1e-6
    use_spitzer: bool = True
    coulomb_log: float = 15.0

    def compute_kappa_parallel(self, T: Array) -> Array:
        """Compute parallel thermal conductivity."""
        if self.use_spitzer:
            # Spitzer: κ_∥ = κ_0 T^(5/2) / ln(Λ)
            T_safe = jnp.maximum(T, 1e-3)  # Avoid T=0
            return self.kappa_parallel_0 * T_safe**2.5 / self.coulomb_log
        return jnp.full_like(T, self.kappa_parallel_0)

    def compute_kappa_perp(self, T: Array) -> Array:
        """Compute perpendicular thermal conductivity."""
        return self.compute_kappa_parallel(T) * self.kappa_perp_ratio

    def compute_heat_flux(self, T: Array, B: Array, dr: float, dz: float,
                         r: Array) -> tuple[Array, Array]:
        """Compute anisotropic heat flux components (q_r, q_z).

        Returns:
            q_r: Radial heat flux (nr, nz)
            q_z: Axial heat flux (nr, nz)
        """
        # Compute temperature gradients
        dT_dr = central_diff_r(T, dr)
        dT_dz = central_diff_z(T, dz)

        # Magnetic field magnitude and unit vector
        B_r, B_phi, B_z = B[:,:,0], B[:,:,1], B[:,:,2]
        B_mag = jnp.sqrt(B_r**2 + B_phi**2 + B_z**2)
        B_mag_safe = jnp.maximum(B_mag, 1e-10)

        b_r = B_r / B_mag_safe
        b_z = B_z / B_mag_safe

        # Parallel gradient: b·∇T
        grad_T_parallel = b_r * dT_dr + b_z * dT_dz

        # Perpendicular gradient: ∇T - (b·∇T) b
        grad_T_perp_r = dT_dr - grad_T_parallel * b_r
        grad_T_perp_z = dT_dz - grad_T_parallel * b_z

        # Conductivities
        kappa_par = self.compute_kappa_parallel(T)
        kappa_perp = self.compute_kappa_perp(T)

        # Heat flux components
        q_r = -kappa_par * grad_T_parallel * b_r - kappa_perp * grad_T_perp_r
        q_z = -kappa_par * grad_T_parallel * b_z - kappa_perp * grad_T_perp_z

        return q_r, q_z
```

### 2.3 Temperature RHS Computation

Add to `ExtendedMHD`:

```python
def compute_temperature_rhs(self, state: State, geometry: Geometry,
                            J_r: Array, J_phi: Array, J_z: Array) -> Array:
    """Compute dT/dt from energy equation.

    dT/dt = (2/3n) * (-p∇·v - ∇·q + ηJ² + Q_aux)
    """
    dr, dz = geometry.dr, geometry.dz
    r = geometry.r_grid
    n = self.halo_model.apply(state.n, geometry)
    T = state.T
    p = n * T  # Pressure from ideal gas law

    # 1. Compressional heating: -p∇·v
    v_r, v_z = state.v[:,:,0], state.v[:,:,2]
    # Cylindrical divergence: (1/r) d(r*v_r)/dr + dv_z/dz
    div_v = cylindrical_divergence_rz(v_r, v_z, dr, dz, r)
    compression = -p * div_v

    # 2. Ohmic heating: ηJ²
    J_squared = J_r**2 + J_phi**2 + J_z**2
    eta = self.resistivity.compute(J_phi)
    ohmic = eta * J_squared

    # 3. Heat conduction: -∇·q
    q_r, q_z = self.thermal.compute_heat_flux(T, state.B, dr, dz, r)
    div_q = cylindrical_divergence_rz(q_r, q_z, dr, dz, r)
    conduction = -div_q

    # 4. Auxiliary heating (placeholder)
    Q_aux = jnp.zeros_like(T)

    # Combine: dT/dt = (2/3n) * (sources)
    n_safe = jnp.maximum(n, self.halo_model.halo_density)
    dT_dt = (2.0 / 3.0) / n_safe * (compression + ohmic + conduction + Q_aux)

    return dT_dt
```

### 2.4 Modified compute_rhs

```python
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    """Compute dB/dt and dT/dt from extended MHD equations."""
    # ... existing B computation ...

    # Compute current (needed for both B and T RHS)
    J_r, J_phi, J_z = self._compute_current(B_r, B_phi, B_z, dr, dz, r)

    # ... existing dB computation ...

    # NEW: Temperature evolution
    dT = self.compute_temperature_rhs(state, geometry, J_r, J_phi, J_z)

    return state.replace(B=dB, T=dT)
```

### 2.5 Semi-Implicit Treatment

The parallel conduction term can be extremely stiff (κ_∥ ~ 10²⁰). Options:

**Option A: Super Time Stepping (STS)**
Use Runge-Kutta-Chebyshev methods that allow larger timesteps for diffusion.

**Option B: Implicit Conduction**
Treat conduction implicitly while keeping other terms explicit:
```
T^{n+1} = T^n + Δt * (explicit_terms) + Δt * L_cond(T^{n+1})
```
Requires solving: `(I - Δt L_cond) T^{n+1} = RHS`

**Option C: Directional Splitting**
Treat parallel conduction implicitly along field lines, perpendicular explicitly.

**Recommended: Start with Option A (STS)** for simplicity, then add implicit if needed.

---

## 3. Numerical Considerations

### 3.1 Cylindrical Divergence

For the divergence of (q_r, q_z) in cylindrical coordinates:

```python
def cylindrical_divergence_rz(f_r: Array, f_z: Array, dr: float, dz: float,
                               r: Array) -> Array:
    """Compute ∇·f = (1/r) d(r*f_r)/dr + df_z/dz in cylindrical coords."""
    # d(r*f_r)/dr
    rf_r = r * f_r
    d_rf_r_dr = central_diff_r(rf_r, dr)

    # df_z/dz
    df_z_dz = central_diff_z(f_z, dz)

    # Handle r=0 with L'Hopital
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    div = jnp.where(
        r > 1e-10,
        d_rf_r_dr / r_safe + df_z_dz,
        2.0 * central_diff_r(f_r, dr) + df_z_dz  # L'Hopital at r=0
    )
    return div
```

### 3.2 Temperature Boundary Conditions

At walls (conducting):
- Dirichlet: `T = T_wall` (fixed wall temperature)
- Neumann: `∂T/∂n = 0` (insulating wall)

At axis (r=0):
- Symmetry: `∂T/∂r = 0`

### 3.3 CFL Condition

The conduction CFL is:
```
Δt < dx² / (2 * κ_∥ / (3/2 n))
```

For typical FRC parameters with Spitzer conductivity, this can be very restrictive (~10⁻¹² s), motivating implicit treatment.

---

## 4. Testing Strategy

### 4.1 Unit Tests

1. **Heat flux decomposition**: Verify q_∥ and q_⊥ are orthogonal/parallel to B
2. **Spitzer scaling**: Check κ ∝ T^(5/2)
3. **Divergence operator**: Verify cylindrical divergence at r=0

### 4.2 Physics Tests

1. **Ohmic heating only**: Uniform J, verify T increases as ηJ²t
2. **Parallel conduction**: Temperature blob aligned with B, verify diffusion along B
3. **Perpendicular conduction**: Temperature blob perpendicular to B, verify slow diffusion
4. **Energy conservation**: Total energy (thermal + magnetic) conserved without sources

### 4.3 Benchmark Tests

Compare with analytical solutions for:
1. 1D heat conduction in slab geometry
2. Cylindrical geometry with uniform B_z

---

## 5. Implementation Tasks

### Task 1: Extend State with Temperature
- Add `T: Array` field to State dataclass
- Update `State.zeros()` to initialize T
- Update pytree registration
- Update existing tests

### Task 2: Implement ThermalTransport Class
- Create `jax_frc/models/energy.py`
- Implement `ThermalTransport` dataclass
- Implement `compute_kappa_parallel/perp`
- Implement `compute_heat_flux`
- Add unit tests

### Task 3: Add Cylindrical Divergence Operator
- Add `cylindrical_divergence_rz` to operators.py
- Handle r=0 singularity with L'Hopital
- Add unit tests

### Task 4: Implement Temperature RHS
- Add `compute_temperature_rhs` to ExtendedMHD
- Integrate compression, Ohmic, conduction terms
- Modify `compute_rhs` to return dT
- Add unit tests

### Task 5: Temperature Boundary Conditions
- Add `apply_temperature_bc` method
- Support Dirichlet and Neumann at walls
- Symmetry at axis
- Add tests

### Task 6: Update Semi-Implicit Solver
- Handle stiff conduction term
- Implement Super Time Stepping or implicit option
- Add stability tests

### Task 7: Integration Tests
- End-to-end simulation with temperature
- Energy conservation check
- Benchmark against analytical solutions

---

## 6. Future Extensions

1. **Two-fluid (Ti, Te)**: Separate ion and electron temperatures with equilibration
2. **Radiation losses**: Add `P_rad = n² Λ(T)` with cooling curve
3. **Ionization energy sink**: For neutral coupling
4. **NBI heating**: Auxiliary heating source from beam injection

---

## References

1. Braginskii, S.I. "Transport Processes in a Plasma" Reviews of Plasma Physics 1, 205 (1965)
2. Sovinec, C.R. et al. "Nonlinear magnetohydrodynamics simulation using high-order finite elements" J. Comput. Phys. 195, 355 (2004)
3. plasma_physics.md - Reference document with FRC model derivations
