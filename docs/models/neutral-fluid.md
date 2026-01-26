# Neutral Fluid Model

Hydrodynamic neutral gas model for plasma-neutral coupling in FRC simulations.

## Overview

The neutral fluid model solves compressible Euler equations for a neutral gas species. It enables simulation of:

- Neutral fill ionization during FRC formation
- Charge exchange momentum drag
- Radiation losses from atomic processes

Based on the Lamy Ridge formulation for FRC formation with mTorr neutral fills.

## Equations

The neutral fluid evolves three conservative quantities:

| Equation | Form |
|----------|------|
| Mass | ∂ρn/∂t + ∇·(ρn vn) = -S_ion + S_rec |
| Momentum | ∂(ρn vn)/∂t + ∇·(ρn vn vn + pn I) = -R_cx |
| Energy | ∂En/∂t + ∇·((En + pn) vn) = -Q_cx |

Source terms couple to plasma equations with opposite signs (mass/momentum/energy conservation).

## Usage

```python
from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
import jax.numpy as jnp

# Create neutral state
neutral = NeutralState(
    rho_n=jnp.ones((64, 128)) * 1e-6,      # Mass density [kg/m³]
    mom_n=jnp.zeros((64, 128, 3)),          # Momentum density [kg/m²/s]
    E_n=jnp.ones((64, 128)) * 1e-3          # Energy density [J/m³]
)

# Access derived quantities
velocity = neutral.v_n       # [m/s]
pressure = neutral.p_n       # [Pa]
temperature = neutral.T_n    # [J]

# Create model and compute RHS
model = NeutralFluid(gamma=5/3)
d_rho, d_mom, d_E = model.compute_flux_divergence(neutral, geometry)

# Apply boundary conditions
neutral = model.apply_boundary_conditions(neutral, geometry, bc_type="reflecting")
```

## NeutralState

Immutable dataclass for neutral fluid variables:

```python
@dataclass(frozen=True)
class NeutralState:
    rho_n: Array  # Mass density [kg/m³], shape (nr, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nr, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nr, nz)
```

### Derived Properties

| Property | Formula | Units |
|----------|---------|-------|
| `v_n` | mom_n / rho_n | m/s |
| `p_n` | (γ-1)(E_n - ½ρn v²) | Pa |
| `T_n` | p_n / (rho_n / m_i) | J |

The state is registered as a JAX pytree for JIT compatibility.

## NeutralFluid Model

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 5/3 | Adiabatic index (monatomic gas) |

### Methods

#### `compute_flux_divergence`

```python
def compute_flux_divergence(
    self, state: NeutralState, geometry: Geometry
) -> Tuple[Array, Array, Array]:
```

Computes -∇·F for Euler equations using HLLE approximate Riemann solver.

Returns `(d_rho, d_mom, d_E)` - the flux divergence terms for each equation.

#### `apply_boundary_conditions`

```python
def apply_boundary_conditions(
    self, state: NeutralState, geometry: Geometry, bc_type: str = "reflecting"
) -> NeutralState:
```

**Boundary types:**

| Type | Description |
|------|-------------|
| `"reflecting"` | Wall with bounce-back (v_normal → -v_normal) |
| `"absorbing"` | Outflow with zero gradient (Neumann) |

**Axis handling (r=0):**
- v_r = 0, v_θ = 0 (symmetry)
- Scalars: zero gradient (Neumann)

## HLLE Riemann Solver

The model uses HLLE (Harten-Lax-van Leer-Einfeldt) for shock-capturing:

```python
@jit
def hlle_flux_1d(rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, gamma=5/3):
    """HLLE approximate Riemann solver for 1D Euler equations."""
```

**Wave speed estimates (Davis method):**
- S_L = min(v_L - c_L, v_R - c_R)
- S_R = max(v_L + c_L, v_R + c_R)

**HLLE flux:**
```
F_HLLE = (S_R·F_L - S_L·F_R + S_L·S_R·(U_R - U_L)) / (S_R - S_L)
```

## Atomic Rates

Source terms for plasma-neutral coupling are provided in `jax_frc.models.atomic_rates`:

### Ionization

```python
from jax_frc.models.atomic_rates import ionization_rate, ionization_rate_coefficient

# Rate coefficient <σv>_ion(Te) using Voronov fit
sigma_v = ionization_rate_coefficient(Te)  # [m³/s]

# Mass ionization rate
S_ion = ionization_rate(Te, ne, rho_n)  # [kg/m³/s]
```

### Recombination

```python
from jax_frc.models.atomic_rates import recombination_rate

S_rec = recombination_rate(Te, ne, ni)  # [kg/m³/s]
```

### Charge Exchange

```python
from jax_frc.models.atomic_rates import charge_exchange_rates

R_cx, Q_cx = charge_exchange_rates(Ti, ni, nn, v_i, v_n)
# R_cx: Momentum transfer [N/m³], shape (nr, nz, 3)
# Q_cx: Energy transfer [W/m³], shape (nr, nz)
```

### Radiation Losses

```python
from jax_frc.models.atomic_rates import total_radiation_loss

P_rad = total_radiation_loss(Te, ne, ni, n_impurity, S_ion, Z_eff)
```

Includes:
- Bremsstrahlung: P_brem = 1.69e-38 · Z² · ne · ni · √Te
- Line radiation: Gaussian cooling curve peaked at ~10 eV
- Ionization energy: 13.6 eV per ionization event

## CFL Condition

Timestep limited by acoustic CFL:

```
dt < CFL · min(dr, dz) / max(|v_n| + c_s)
```

where c_s = √(γ p / ρ) is the sound speed.

## Example: Neutral Fill Ionization

```python
import jax.numpy as jnp
from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
from jax_frc.models.atomic_rates import ionization_rate

# Initial neutral fill (1 mTorr hydrogen at room temp)
n_fill = 3.2e19  # m⁻³ at 1 mTorr
T_neutral = 0.025 * QE  # 0.025 eV (room temp)
rho_n = n_fill * MI

neutral = NeutralState(
    rho_n=jnp.full((nr, nz), rho_n),
    mom_n=jnp.zeros((nr, nz, 3)),
    E_n=jnp.full((nr, nz), 1.5 * n_fill * T_neutral)
)

# Compute ionization during simulation
S_ion = ionization_rate(plasma_state.Te, plasma_state.ne, neutral.rho_n)

# Update neutral density: d(rho_n)/dt includes -S_ion
```

## Limitations

1. **No neutral-neutral collisions** - Valid for low neutral densities
2. **Single species** - Hydrogen only (extension to D, T, He possible)
3. **No wall recycling** - Neutrals absorbed at boundaries
4. **Cold neutral approximation** - T_n << T_i assumed for charge exchange

## Related

- [Resistive MHD](resistive-mhd.md) - Base plasma model
- [Extended MHD](extended-mhd.md) - With Hall physics
