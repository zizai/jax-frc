# Burning Plasma Model Design

## Overview

A new physics model for simulating nuclear burn dynamics in FRC plasmas, with full self-consistent coupling between fusion reactions, fuel evolution, anomalous transport, and direct induction energy recovery.

## Requirements

- **Multi-fuel support:** D-T, D-D, D-³He reactions
- **Instant thermalization:** Fusion power deposited directly to thermal plasma
- **Direct induction:** Energy recovery from changing magnetic flux
- **Full coupling:** Burn affects plasma evolution, plasma affects burn rate
- **Anomalous transport:** Configurable particle and energy diffusion
- **2D axisymmetric:** (r,z) geometry capturing FRC elongation and X-points

## Architecture

Composition-based design with separate physics modules:

```
BurningPlasmaModel
├── MHDCore          # Field evolution (B, v, p) - reuses existing resistive MHD
├── BurnPhysics      # Fusion reaction rates and power for D-T, D-D, D-³He
├── SpeciesTracker   # Fuel/ash densities (D, T, ³He, ⁴He, p)
├── TransportModel   # Anomalous particle and energy diffusion
└── DirectConversion # Geometry-aware direct induction energy recovery
```

### State Structure

```python
@dataclass
class BurningPlasmaState:
    mhd: MHDState           # B, v, p, ρ (existing)
    species: SpeciesState   # n_D, n_T, n_He3, n_He4, n_p
    burn: BurnState         # P_fusion, reaction rates, Q_alpha
    conversion: ConversionState  # P_electric, η_effective
```

### Coupling Flow Per Timestep

1. `BurnPhysics` computes reaction rates from current n, T
2. `SpeciesTracker` updates fuel/ash densities (depletion + transport)
3. `TransportModel` computes diffusive fluxes for particles and energy
4. `DirectConversion` computes recovered electrical power from flux change
5. `MHDCore` advances fields with burn power as energy source term
6. State is updated atomically (JAX immutable style)

## Module Specifications

### BurnPhysics

Computes fusion reaction rates and power deposition.

**Reaction rate calculation:**
```python
@dataclass
class BurnPhysics:
    fuels: tuple[str, ...]  # e.g., ("DT", "DD", "DHe3")

    def reaction_rate(self, n1, n2, T, reaction: str) -> jax.Array:
        """Volumetric reaction rate [reactions/m³/s]."""
        sigma_v = self.reactivity(T, reaction)  # <σv> from fits
        kronecker = 1.0 if reaction in ("DD",) else 0.0
        return n1 * n2 * sigma_v / (1.0 + kronecker)
```

**Supported reactions:**

| Reaction | Products | Energy | Charged fraction |
|----------|----------|--------|------------------|
| D + T → ⁴He + n | 3.5 MeV α, 14.1 MeV n | 17.6 MeV | 20% |
| D + D → T + p | 1.01 MeV T, 3.02 MeV p | 4.03 MeV | 100% |
| D + D → ³He + n | 0.82 MeV ³He, 2.45 MeV n | 3.27 MeV | 25% |
| D + ³He → ⁴He + p | 3.6 MeV α, 14.7 MeV p | 18.3 MeV | 100% |

**Reactivity fits:** Bosch-Hale parameterization (NF 1992) for ⟨σv⟩(T), valid 0.2-100 keV.

**Power outputs:**
- `P_fusion`: total fusion power density [W/m³]
- `P_charged`: power in charged products
- `P_neutron`: power in neutrons (escapes plasma)
- `P_alpha`: power deposited to plasma (instant thermalization)

### SpeciesTracker

Tracks fuel consumption and ash accumulation.

**Species tracked:**
```python
@dataclass
class SpeciesState:
    n_D: jax.Array    # Deuterium density [m⁻³]
    n_T: jax.Array    # Tritium density [m⁻³]
    n_He3: jax.Array  # Helium-3 density [m⁻³]
    n_He4: jax.Array  # Helium-4 (ash) density [m⁻³]
    n_p: jax.Array    # Proton density [m⁻³]
```

**Evolution equation:**
```
∂nₛ/∂t = -∇·Γₛ + Sₛ_burn + Sₛ_source
```

**Burn source terms:**
```python
def burn_sources(self, rates: ReactionRates) -> dict[str, jax.Array]:
    return {
        'D':   -rates.DT - 2*rates.DD - rates.DHe3,
        'T':   -rates.DT + rates.DD_T,
        'He3': -rates.DHe3 + rates.DD_He3,
        'He4': +rates.DT + rates.DHe3,
        'p':   +rates.DD_T + rates.DHe3,
    }
```

**Quasi-neutrality:** `n_e = n_D + n_T + n_He3 + 2*n_He4 + n_p`

### TransportModel

Computes anomalous particle and energy fluxes.

**Transport coefficients:**
```python
@dataclass
class TransportModel:
    D_particle: float | jax.Array  # Particle diffusivity [m²/s]
    χ_e: float | jax.Array         # Electron thermal diffusivity [m²/s]
    χ_i: float | jax.Array         # Ion thermal diffusivity [m²/s]
    v_pinch: float | jax.Array     # Inward pinch velocity [m/s]
```

**Particle flux:**
```
Γₛ = -D_particle ∇nₛ + nₛ v_pinch
```

**Energy flux:**
```
q_e = -n_e χ_e ∇T_e
q_i = -n_i χ_i ∇T_i
```

**Boundary conditions:** Zero flux at symmetry axis; configurable at outer boundary.

### DirectConversion

Computes electrical power recovery via direct induction from time-varying magnetic flux.

**Physics:** As fusion heats the plasma, the FRC expands against the external field. Changing magnetic flux induces voltage in pickup coils:
```
V_induced = -dΨ/dt
```

**Implementation:**
```python
@dataclass
class DirectConversion:
    coil_turns: int
    coil_radius: float
    circuit_resistance: float
    coupling_efficiency: float

    def compute_power(self, B_old: jax.Array, B_new: jax.Array,
                      dt: float, geom: Geometry) -> ConversionState:
        Psi_old = self.coil_turns * flux_integral(B_old, geom, self.coil_radius)
        Psi_new = self.coil_turns * flux_integral(B_new, geom, self.coil_radius)

        dPsi_dt = (Psi_new - Psi_old) / dt
        V_induced = -dPsi_dt * self.coupling_efficiency
        P_electric = V_induced**2 / (4 * self.circuit_resistance)

        return ConversionState(P_electric=P_electric, V_induced=V_induced, dPsi_dt=dPsi_dt)
```

**Back-reaction:** Induced current creates back-EMF opposing flux change, coupled to MHD as effective drag on plasma expansion.

### BurningPlasmaModel Orchestration

```python
@dataclass
class BurningPlasmaModel:
    mhd_core: MHDCore
    burn: BurnPhysics
    species: SpeciesTracker
    transport: TransportModel
    conversion: DirectConversion

    @jax.jit
    def step(self, state: BurningPlasmaState, dt: float,
             geom: Geometry) -> BurningPlasmaState:
        # 1. Compute fusion reaction rates from current n, T
        T = temperature_from_pressure(state.mhd.p, state.species)
        rates = self.burn.reaction_rates(state.species, T)

        # 2. Compute source terms
        burn_sources = self.burn.power_sources(rates)
        species_sources = self.species.burn_sources(rates)
        transport_fluxes = self.transport.compute_fluxes(state, geom)

        # 3. Update species densities (burn + transport)
        new_species = self.species.advance(
            state.species, species_sources, transport_fluxes, dt, geom
        )

        # 4. Advance MHD with alpha heating source
        energy_source = burn_sources.P_alpha - transport_fluxes.energy_divergence
        new_mhd = self.mhd_core.step(state.mhd, energy_source, dt, geom)

        # 5. Compute direct induction power from B-field change
        new_conversion = self.conversion.compute_power(
            state.mhd.B, new_mhd.B, dt, geom
        )

        # 6. Apply back-reaction to MHD (energy extracted)
        new_mhd = apply_induction_drag(new_mhd, new_conversion, dt)

        return BurningPlasmaState(
            mhd=new_mhd, species=new_species,
            burn=BurnState(rates=rates, **burn_sources),
            conversion=new_conversion
        )
```

## Testing Strategy

### Unit Tests (fast, mocked)

| Test | Verifies |
|------|----------|
| `test_reactivity_fits` | Bosch-Hale ⟨σv⟩ matches published values |
| `test_burn_sources_conservation` | Total particles conserved |
| `test_species_quasi_neutrality` | n_e computed correctly |
| `test_transport_flux_symmetry` | Zero flux at axis |
| `test_induction_voltage_sign` | Expanding plasma → positive voltage |

### Invariant Tests

```python
def test_energy_conservation(state_old, state_new):
    P_fusion = state_new.burn.P_fusion
    P_alpha = state_new.burn.P_alpha
    P_neutron = state_new.burn.P_neutron
    P_electric = state_new.conversion.P_electric

    assert_close(P_fusion, P_alpha + P_neutron + P_electric, rtol=1e-10)
```

### Physics Validation (slow)

- D-T ignition threshold at expected n·τ·T
- Fuel burnup rate matches reaction rate integral
- Lawson criterion comparison against published results

## File Structure

```
jax_frc/
├── models/
│   ├── burning_plasma.py      # BurningPlasmaModel, BurningPlasmaState
│   └── ...
├── burn/
│   ├── __init__.py
│   ├── physics.py             # BurnPhysics, ReactionRates, reactivity fits
│   ├── species.py             # SpeciesTracker, SpeciesState
│   └── conversion.py          # DirectConversion, ConversionState
├── transport/
│   ├── __init__.py
│   └── anomalous.py           # TransportModel, TransportFluxes
├── constants.py               # Add Bosch-Hale coefficients, reaction energies
└── ...

tests/
├── test_burn_physics.py
├── test_species_tracker.py
├── test_direct_conversion.py
├── test_transport.py
├── test_burning_plasma.py
└── ...

docs/
└── models/
    └── burning-plasma.md
```

## Dependencies

- Reuses `MHDCore` from existing `models/resistive_mhd.py`
- No external dependencies beyond JAX and existing `jax_frc` utilities
