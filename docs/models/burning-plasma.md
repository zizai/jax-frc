# Burning Plasma Model

**File:** [burning_plasma.py](../../jax_frc/models/burning_plasma.py)

**Physics Level:** Multi-physics (MHD + Nuclear Burn + Transport)

Multi-fuel nuclear burn model with anomalous transport and direct induction energy recovery.

## Overview

The burning plasma model combines:
- **MHD core**: Resistive MHD for field evolution
- **Burn physics**: D-T, D-D, D-He3 fusion reactions
- **Species tracking**: Fuel depletion and ash accumulation
- **Anomalous transport**: Configurable particle and energy diffusion
- **Direct conversion**: Induction-based energy recovery

## Best For

- Burning plasma simulations
- Q-factor studies
- Energy balance analysis
- Direct conversion optimization

## Computational Cost

Medium (MHD + species tracking + transport)

## Configuration

```yaml
model:
  type: burning_plasma
  fuels: [DT, DD]
  mhd:
    resistivity:
      type: spitzer
      eta_0: 1e-6
  transport:
    D_particle: 1.0   # m^2/s
    chi_e: 5.0        # m^2/s
    chi_i: 2.0        # m^2/s
  direct_conversion:
    coil_turns: 100
    coil_radius: 0.6  # m
    circuit_resistance: 0.1  # Ohm
    coupling_efficiency: 0.9
```

## State Variables

| Field | Description | Units |
|-------|-------------|-------|
| `species.n_D` | Deuterium density | m^-3 |
| `species.n_T` | Tritium density | m^-3 |
| `species.n_He4` | Helium-4 (ash) density | m^-3 |
| `power.P_fusion` | Total fusion power density | W/m^3 |
| `power.P_alpha` | Alpha heating power | W/m^3 |
| `conversion.P_electric` | Recovered electrical power | W |

## Physics

### Reactivity

Uses Bosch-Hale parameterization (NF 1992) for sigma-v(T).

### Transport

Anomalous diffusive transport:
- Particle flux: Gamma = -D * grad(n) + n * v_pinch
- Energy flux: q = -n * chi * grad(T)

### Direct Conversion

Power recovered via magnetic induction:
```
V = -N * d(Psi)/dt * eta_coupling
P = V^2 / (4R)
```

## Usage

### Factory Method

```python
from jax_frc.models import PhysicsModel

model = PhysicsModel.create({
    "type": "burning_plasma",
    "fuels": ["DT"],
    "transport": {
        "D_particle": 1.0,
        "chi_e": 5.0,
        "chi_i": 2.0,
    },
    "direct_conversion": {
        "coil_turns": 100,
        "coil_radius": 0.6,
    }
})
```

### Direct Construction

```python
from jax_frc.models import BurningPlasmaModel, BurningPlasmaState
from jax_frc.models import ResistiveMHD
from jax_frc.burn import BurnPhysics, SpeciesTracker, DirectConversion
from jax_frc.transport import TransportModel

model = BurningPlasmaModel(
    mhd_core=ResistiveMHD.from_config({"resistivity": {"type": "spitzer"}}),
    burn=BurnPhysics(fuels=("DT",)),
    species_tracker=SpeciesTracker(),
    transport=TransportModel(D_particle=1.0, chi_e=5.0, chi_i=2.0),
    conversion=DirectConversion(coil_turns=100, coil_radius=0.6),
)
```

## Implementation Details

### BurningPlasmaState

The composite state contains:
- `mhd`: MHD state (B, v, p, psi, etc.)
- `species`: Fuel and ash densities (SpeciesState)
- `rates`: Current reaction rates (ReactionRates)
- `power`: Current power sources (PowerSources)
- `conversion`: Direct conversion state (ConversionState)

### Time Stepping

The `step()` method advances the state:
1. Compute fusion reaction rates from species densities and temperature
2. Compute power sources (alpha heating, neutron power)
3. Compute burn source terms for species evolution
4. Compute transport fluxes and divergences
5. Update species densities
6. Compute direct conversion power from B-field changes
7. Update MHD state (with alpha heating feedback)

### JAX Pytree Registration

BurningPlasmaState is registered as a JAX pytree for automatic differentiation and JIT compilation compatibility.
