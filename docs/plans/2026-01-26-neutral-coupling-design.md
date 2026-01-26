# Neutral-Fluid Coupling Design (Lamy Ridge-style)

**Date**: 2026-01-26
**Status**: Approved
**Goal**: Add neutral fluid + atomic physics to resistive MHD for FRC formation simulations

## Overview

Extends resistive MHD with a second fluid species (neutrals) that exchanges mass, momentum, and energy with the plasma through atomic processes. This enables realistic FRC formation simulations with mTorr neutral fills.

Based on Lamy Ridge formulation: 2D resistive MHD + energy equation + neutral fluid + ionization + charge exchange + radiation losses.

## Architecture

```
jax_frc/
├── models/
│   ├── resistive_mhd.py      (extended with neutral coupling)
│   ├── neutral_fluid.py      (NEW: neutral Euler equations)
│   └── atomic_rates.py       (NEW: ionization, CX, radiation)
├── configurations/
│   └── neutral_validation.py (NEW: analytic test cases)
└── tests/
    └── test_neutral_coupling.py (NEW)
```

## Equation System

### Plasma (existing + sources)

| Equation | Form |
|----------|------|
| Mass | ∂ρ/∂t + ∇·(ρv) = +S_ion - S_rec |
| Momentum | ∂(ρv)/∂t + ∇·(ρvv + pI) = J×B + R_ion + R_cx |
| Energy | ∂E/∂t + ∇·((E+p)v) = ηJ² - P_rad - P_ion + Q_cx |
| Induction | ∂ψ/∂t = (η/μ₀)Δ*ψ - v·∇ψ |

### Neutral fluid (new)

| Equation | Form |
|----------|------|
| Mass | ∂ρn/∂t + ∇·(ρn·vn) = -S_ion + S_rec |
| Momentum | ∂(ρn·vn)/∂t + ∇·(ρn·vn·vn + pn·I) = -R_ion - R_cx |
| Energy | ∂En/∂t + ∇·((En + pn)·vn) = -Q_cx |

## Component 1: Atomic Rate Coefficients

**File:** `jax_frc/models/atomic_rates.py`

### Ionization (electron impact: H + e → H⁺ + 2e)

```python
@jit
def ionization_rate_coefficient(Te: Array) -> Array:
    """Voronov fit for hydrogen ionization <σv>_ion(Te) [m³/s].

    <σv> = A * (1 + P*sqrt(U)) * U^K * exp(-U) / (X + U)
    where U = E_ion / Te, E_ion = 13.6 eV
    """
    E_ion = 13.6 * QE  # Ionization energy in Joules
    U = E_ion / Te
    A, P, K, X = 2.91e-14, 0.0, 0.39, 0.232  # Voronov coefficients
    return A * (1 + P * jnp.sqrt(U)) * U**K * jnp.exp(-U) / (X + U)

@jit
def ionization_rate(Te: Array, ne: Array, rho_n: Array) -> Array:
    """Mass ionization rate S_ion [kg/m³/s]."""
    nn = rho_n / MI  # Neutral number density
    sigma_v = ionization_rate_coefficient(Te)
    return MI * ne * nn * sigma_v
```

### Recombination (radiative: H⁺ + e → H + hν)

```python
@jit
def recombination_rate_coefficient(Te: Array) -> Array:
    """Radiative recombination <σv>_rec(Te) [m³/s].

    Approximate: <σv>_rec ≈ 2.6e-19 * (13.6 eV / Te)^0.7
    """
    Te_eV = Te / QE
    return 2.6e-19 * (13.6 / jnp.maximum(Te_eV, 0.1))**0.7

@jit
def recombination_rate(Te: Array, ne: Array, ni: Array) -> Array:
    """Mass recombination rate S_rec [kg/m³/s]."""
    sigma_v = recombination_rate_coefficient(Te)
    return MI * ne * ni * sigma_v
```

### Charge Exchange (H⁺ + H → H + H⁺)

```python
@jit
def charge_exchange_cross_section(Ti: Array) -> Array:
    """CX cross-section σ_cx(Ti) [m²].

    Nearly constant ~3e-19 m² for Ti < 10 keV.
    """
    return 3.0e-19 * jnp.ones_like(Ti)

@jit
def charge_exchange_rates(
    Ti: Array, ni: Array, nn: Array,
    v_i: Array, v_n: Array
) -> Tuple[Array, Array]:
    """Charge exchange momentum and energy transfer.

    Returns:
        R_cx: Momentum transfer [N/m³] (vector, shape matches v_i)
        Q_cx: Energy transfer [W/m³] (scalar field)
    """
    v_thermal = jnp.sqrt(8 * Ti / (jnp.pi * MI))  # Mean thermal speed
    sigma = charge_exchange_cross_section(Ti)
    nu_cx = nn * sigma * v_thermal  # CX collision frequency

    # Momentum transfer: R_cx = m_i * n_i * nu_cx * (v_i - v_n)
    R_cx = MI * ni * nu_cx * (v_i - v_n)

    # Energy transfer: Q_cx = (3/2) * n_i * nu_cx * (T_i - T_n)
    # Simplified: assume T_n << T_i for cold neutrals
    Q_cx = 1.5 * ni * nu_cx * Ti

    return R_cx, Q_cx
```

### Radiation Losses

```python
@jit
def bremsstrahlung_loss(Te: Array, ne: Array, ni: Array, Z_eff: float = 1.0) -> Array:
    """Bremsstrahlung power loss P_brem [W/m³].

    P_brem = 1.69e-38 * Z_eff² * ne * ni * sqrt(Te_eV)
    """
    Te_eV = Te / QE
    return 1.69e-38 * Z_eff**2 * ne * ni * jnp.sqrt(jnp.maximum(Te_eV, 0.1))

@jit
def line_radiation_loss(Te: Array, ne: Array, n_impurity: Array) -> Array:
    """Line radiation from impurities using coronal equilibrium.

    Uses piecewise fit to cooling curve L(Te).
    P_line = ne * n_imp * L(Te)
    """
    Te_eV = Te / QE
    # Simplified cooling curve for carbon impurity (dominant in most experiments)
    # Peak around 10 eV, drops at higher Te
    L_cool = 1e-31 * jnp.exp(-((jnp.log10(Te_eV) - 1.0) / 0.5)**2)
    return ne * n_impurity * L_cool

@jit
def ionization_energy_loss(S_ion: Array) -> Array:
    """Energy sink from ionization events.

    Each ionization costs E_ion = 13.6 eV.
    P_ion = S_ion * E_ion / m_i = (ionizations/s/m³) * E_ion
    """
    E_ion = 13.6 * QE
    return S_ion * E_ion / MI

@jit
def total_radiation_loss(
    Te: Array, ne: Array, ni: Array,
    n_impurity: Array, S_ion: Array,
    Z_eff: float = 1.0
) -> Array:
    """Total radiation sink for energy equation [W/m³]."""
    P_brem = bremsstrahlung_loss(Te, ne, ni, Z_eff)
    P_line = line_radiation_loss(Te, ne, n_impurity)
    P_ion = ionization_energy_loss(S_ion)
    return P_brem + P_line + P_ion
```

## Component 2: Neutral Fluid

**File:** `jax_frc/models/neutral_fluid.py`

### State Container

```python
@dataclass
class NeutralState:
    """Neutral fluid state variables."""
    rho_n: Array    # Mass density [kg/m³]
    mom_n: Array    # Momentum density (nr, nz, 3) [kg/m²/s]
    E_n: Array      # Total energy density [J/m³]

    @property
    def v_n(self) -> Array:
        """Velocity [m/s]."""
        return self.mom_n / jnp.maximum(self.rho_n[..., None], 1e-20)

    @property
    def p_n(self) -> Array:
        """Pressure [Pa] from ideal gas EOS."""
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / jnp.maximum(self.rho_n, 1e-20)
        return (GAMMA - 1) * (self.E_n - ke)

    @property
    def T_n(self) -> Array:
        """Temperature [J] = p / n."""
        n_n = self.rho_n / MI
        return self.p_n / jnp.maximum(n_n, 1e-10)
```

### HLLE Riemann Solver

```python
@jit
def hlle_flux(rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, gamma: float = 5/3):
    """HLLE approximate Riemann solver for Euler equations.

    Returns numerical flux at cell interface.
    """
    # Sound speeds
    c_L = jnp.sqrt(gamma * p_L / jnp.maximum(rho_L, 1e-20))
    c_R = jnp.sqrt(gamma * p_R / jnp.maximum(rho_R, 1e-20))

    # Wave speed estimates (Davis)
    S_L = jnp.minimum(v_L - c_L, v_R - c_R)
    S_R = jnp.maximum(v_L + c_L, v_R + c_R)

    # Fluxes
    F_L = euler_flux(rho_L, v_L, p_L, E_L)
    F_R = euler_flux(rho_R, v_R, p_R, E_R)
    U_L = jnp.array([rho_L, rho_L * v_L, E_L])
    U_R = jnp.array([rho_R, rho_R * v_R, E_R])

    # HLLE flux
    F_hlle = jnp.where(
        S_L >= 0, F_L,
        jnp.where(
            S_R <= 0, F_R,
            (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
        )
    )
    return F_hlle
```

### Neutral Fluid Model

```python
@dataclass
class NeutralFluid:
    """Hydrodynamic neutral species."""

    gamma: float = 5/3  # Monatomic gas

    def compute_rhs(
        self, neutral: NeutralState, geometry: Geometry,
        S_ion: Array = None, S_rec: Array = None,
        R_cx: Array = None, Q_cx: Array = None
    ) -> NeutralState:
        """Compute d(neutral)/dt from Euler equations + sources."""
        dr, dz = geometry.dr, geometry.dz

        # Euler fluxes using HLLE
        d_rho_flux = self._mass_flux_divergence(neutral, dr, dz)
        d_mom_flux = self._momentum_flux_divergence(neutral, dr, dz)
        d_E_flux = self._energy_flux_divergence(neutral, dr, dz)

        # Add source terms (if provided)
        d_rho = d_rho_flux
        d_mom = d_mom_flux
        d_E = d_E_flux

        if S_ion is not None:
            d_rho = d_rho - S_ion + S_rec
        if R_cx is not None:
            d_mom = d_mom - R_cx
        if Q_cx is not None:
            d_E = d_E - Q_cx

        return NeutralState(d_rho, d_mom, d_E)

    def apply_boundary_conditions(
        self, neutral: NeutralState, geometry: Geometry,
        bc_type: str = "reflecting"
    ) -> NeutralState:
        """Apply boundary conditions to neutral state."""
        # Axis (r=0): symmetry
        # Outer wall: reflecting or absorbing
        # Axial ends: reflecting or outflow
        ...
```

## Component 3: Coupled System

**File:** `jax_frc/models/resistive_mhd.py` (extended)

### Combined State

```python
@dataclass
class CoupledState:
    """Combined plasma + neutral state."""
    # Plasma fields
    psi: Array          # Poloidal flux [Wb]
    rho: Array          # Plasma mass density [kg/m³]
    mom: Array          # Plasma momentum density [kg/m²/s]
    E: Array            # Plasma total energy density [J/m³]

    # Neutral fields
    neutral: NeutralState

    # Derived quantities (cached)
    @property
    def ne(self) -> Array:
        """Electron density [m⁻³]."""
        return self.rho / MI  # Quasi-neutrality

    @property
    def Te(self) -> Array:
        """Electron temperature [J]."""
        # Assume Te = Ti for single-temperature model
        p = self.pressure
        return p / (2 * self.ne)  # p = n_e * T_e + n_i * T_i = 2 n T

    @property
    def pressure(self) -> Array:
        """Total pressure [Pa]."""
        ke = 0.5 * jnp.sum(self.mom**2, axis=-1) / jnp.maximum(self.rho, 1e-20)
        B2 = ...  # Compute from psi
        return (GAMMA - 1) * (self.E - ke - B2 / (2 * MU0))
```

### Extended Physics Model

```python
@dataclass
class ResistiveMHDWithNeutrals(PhysicsModel):
    """Resistive MHD coupled to neutral fluid (Lamy Ridge-style)."""

    resistivity: ResistivityModel
    neutrals: NeutralFluid
    include_radiation: bool = True
    impurity_fraction: float = 0.0  # n_imp / n_e

    def compute_rhs(self, state: CoupledState, geometry: Geometry) -> CoupledState:
        """Compute time derivatives for coupled system."""

        # 1. Compute atomic rates
        Te = state.Te
        ne = state.ne
        ni = ne  # Quasi-neutrality
        nn = state.neutral.rho_n / MI
        n_imp = self.impurity_fraction * ne

        S_ion = ionization_rate(Te, ne, state.neutral.rho_n)
        S_rec = recombination_rate(Te, ne, ni)
        R_cx, Q_cx = charge_exchange_rates(
            Te, ni, nn, state.v, state.neutral.v_n
        )
        P_rad = total_radiation_loss(Te, ne, ni, n_imp, S_ion) if self.include_radiation else 0

        # 2. Plasma equations
        d_psi = self._induction_rhs(state, geometry)
        d_rho = self._continuity_rhs(state, geometry) + S_ion - S_rec
        d_mom = self._momentum_rhs(state, geometry) + R_cx
        d_E = self._energy_rhs(state, geometry) - P_rad + Q_cx

        # 3. Neutral equations (with opposite source signs)
        d_neutral = self.neutrals.compute_rhs(
            state.neutral, geometry,
            S_ion=S_ion, S_rec=S_rec, R_cx=R_cx, Q_cx=Q_cx
        )

        return CoupledState(d_psi, d_rho, d_mom, d_E, d_neutral)

    def compute_stable_dt(self, state: CoupledState, geometry: Geometry) -> float:
        """CFL condition for coupled system."""
        # Plasma CFL (existing)
        dt_plasma = super().compute_stable_dt(state, geometry)

        # Neutral CFL: dt < dx / (|v_n| + c_s)
        c_s = jnp.sqrt(self.neutrals.gamma * state.neutral.p_n /
                       jnp.maximum(state.neutral.rho_n, 1e-20))
        v_n_mag = jnp.sqrt(jnp.sum(state.neutral.v_n**2, axis=-1))
        dt_neutral = 0.4 * jnp.minimum(geometry.dr, geometry.dz) / jnp.max(v_n_mag + c_s)

        return jnp.minimum(dt_plasma, dt_neutral)
```

### Operator Splitting for Stiff Sources

```python
def step_with_splitting(
    state: CoupledState, dt: float,
    model: ResistiveMHDWithNeutrals, geometry: Geometry
) -> CoupledState:
    """Strang splitting for stiff atomic sources.

    1. Half-step: advection (explicit)
    2. Full-step: atomic sources (implicit or subcycled)
    3. Half-step: advection (explicit)
    """
    # Half advection step
    state = advection_step(state, dt/2, model, geometry)

    # Full atomic source step (subcycled for stiffness)
    state = atomic_source_step(state, dt, model, geometry)

    # Half advection step
    state = advection_step(state, dt/2, model, geometry)

    return state

def atomic_source_step(
    state: CoupledState, dt: float,
    model: ResistiveMHDWithNeutrals, geometry: Geometry,
    n_subcycles: int = 10
) -> CoupledState:
    """Subcycled integration of stiff atomic sources."""
    dt_sub = dt / n_subcycles

    def subcycle_body(i, s):
        # Compute rates at current state
        Te, ne, ni = s.Te, s.ne, s.ne
        nn = s.neutral.rho_n / MI

        S_ion = ionization_rate(Te, ne, s.neutral.rho_n)
        S_rec = recombination_rate(Te, ne, ni)
        R_cx, Q_cx = charge_exchange_rates(Te, ni, nn, s.v, s.neutral.v_n)
        P_rad = total_radiation_loss(...)

        # Update densities and energies
        new_rho = s.rho + dt_sub * (S_ion - S_rec)
        new_rho_n = s.neutral.rho_n + dt_sub * (-S_ion + S_rec)
        new_E = s.E + dt_sub * (-P_rad + Q_cx)
        ...
        return updated_state

    return lax.fori_loop(0, n_subcycles, subcycle_body, state)
```

## Component 4: Configuration for YAML

**File:** `jax_frc/config/loader.py` (extended)

```yaml
# Example: FRC formation with neutrals
model:
  type: resistive_mhd_neutrals
  resistivity:
    type: chodura
    eta_0: 1e-6
    eta_anom: 1e-3
    threshold: 1e4
  neutrals:
    gamma: 1.667
    initial_density: 1e19  # m^-3 (mTorr fill)
    initial_temperature: 0.025  # eV (room temp)
  radiation:
    enabled: true
    impurity_fraction: 0.01  # 1% carbon
```

## Testing Strategy

### Unit Tests (`tests/test_atomic_rates.py`)

```python
def test_ionization_rate_peak():
    """Ionization rate peaks around 50-100 eV for hydrogen."""

def test_recombination_dominates_low_Te():
    """Recombination > ionization below ~1 eV."""

def test_radiation_positive():
    """All radiation terms are positive (energy sinks)."""

def test_cx_momentum_conservation():
    """R_cx_plasma + R_cx_neutral = 0 (Newton's 3rd law)."""
```

### Conservation Tests (`tests/test_neutral_coupling.py`)

```python
def test_total_mass_conservation():
    """Total mass (plasma + neutral) conserved during ionization."""
    initial_mass = integrate(state.rho + state.neutral.rho_n)
    # Run simulation
    final_mass = integrate(final_state.rho + final_state.neutral.rho_n)
    assert abs(final_mass - initial_mass) / initial_mass < 1e-10

def test_momentum_conservation_isolated():
    """Total momentum conserved with no external forces."""
```

### Analytic Validation (`tests/test_neutral_validation.py`)

```python
class IonizationFrontTest:
    """1D ionization front into uniform neutral gas.

    For constant ionization rate S = ne * nn * <σv>:
    Front position: x_f(t) = (S / rho_n0) * t
    """

class RadiativeCoolingTest:
    """Uniform plasma cooling by radiation.

    dT/dt = -P_rad / (3/2 * n)
    For bremsstrahlung: T(t) = T0 / (1 + t/tau)^2
    where tau = 3 n / (2 * 1.69e-38 * n^2 * sqrt(T0))
    """
```

### Integration Test

```python
def test_frc_formation_neutral_burnout():
    """FRC formation with neutral ionization.

    Initial: mTorr hydrogen fill, external theta-pinch field
    Expected:
    - Neutrals ionize within ~10 μs
    - Plasma heats to ~100 eV
    - Field reversal forms at magnetic axis
    """
```

## Implementation Order

### Phase 1: Atomic Rates (Days 1-2)
1. Create `jax_frc/models/atomic_rates.py`
2. Implement all rate coefficients with JIT
3. Unit tests for rate scaling and limits

### Phase 2: Neutral Fluid (Days 3-4)
1. Create `jax_frc/models/neutral_fluid.py`
2. Implement `NeutralState` dataclass
3. Implement HLLE Riemann solver
4. Implement flux divergence operators
5. Test pure neutral advection

### Phase 3: Coupled System (Days 5-7)
1. Create `CoupledState`
2. Extend `ResistiveMHD` → `ResistiveMHDWithNeutrals`
3. Implement bidirectional source coupling
4. Implement operator splitting
5. Conservation tests

### Phase 4: Validation (Days 8-10)
1. Ionization front analytic test
2. Radiative cooling test
3. FRC formation integration test
4. Documentation

## Expected Outcomes

- Realistic neutral burnout during FRC formation
- Charge exchange effects on edge momentum
- Radiation-limited temperature in core
- Foundation for future impurity transport

## Future Extensions

- Multi-species neutrals (D, T, He)
- Impurity transport (carbon, oxygen)
- Neutral beam injection sources
- Wall recycling models
