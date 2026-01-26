# Burn Module

Fusion reaction physics for burning plasma simulations.

## Overview

The `jax_frc/burn/` module provides:
- **Reaction rates**: Bosch-Hale parameterization for D-T, D-D, D-He3
- **Species tracking**: Fuel depletion and ash accumulation
- **Direct conversion**: Induction-based energy recovery

## Physics

### Fusion Reactions

| Reaction | Products | Energy | Charged Fraction |
|----------|----------|--------|------------------|
| D + T → α + n | 3.5 MeV α, 14.1 MeV n | 17.6 MeV | 20% |
| D + D → T + p | 1.0 MeV T, 3.0 MeV p | 4.0 MeV | 100% |
| D + D → ³He + n | 0.8 MeV ³He, 2.5 MeV n | 3.3 MeV | 24% |
| D + ³He → α + p | 3.6 MeV α, 14.7 MeV p | 18.3 MeV | 100% |

### Reactivity

Uses Bosch-Hale parameterization (Nuclear Fusion 1992):

$$\langle \sigma v \rangle = C_1 \theta \sqrt{\frac{\xi}{m_r c^2 T^3}} \exp(-3\xi)$$

Where $\theta$ and $\xi$ are temperature-dependent functions with fitted coefficients.

### Power Balance

- **Fusion power**: $P_{fus} = n_1 n_2 \langle \sigma v \rangle E_{fus}$
- **Alpha heating**: $P_\alpha = f_{charged} \cdot P_{fus}$
- **Q-factor**: $Q = P_{fus} / P_{input}$

## Implementation

### Module Structure

```
burn/
├── physics.py      # reactivity(), ReactionRates, PowerSources
├── species.py      # SpeciesState, SpeciesTracker
└── conversion.py   # DirectConversion
```

### Key Functions

**`reactivity(T_keV, reaction)`** - Compute $\langle \sigma v \rangle$

```python
from jax_frc.burn import reactivity

# D-T reactivity at 10 keV
sigma_v = reactivity(10.0, "DT")  # Returns ~1e-22 m³/s
```

**`compute_reaction_rates(species, T_keV)`** - Volumetric rates

```python
from jax_frc.burn import compute_reaction_rates

rates = compute_reaction_rates(species_state, T_keV)
# rates.DT, rates.DD_T, rates.DD_HE3, rates.DHE3 in reactions/m³/s
```

**`compute_power_sources(rates)`** - Power from each channel

```python
from jax_frc.burn import compute_power_sources

power = compute_power_sources(rates)
# power.P_fusion, power.P_alpha, power.P_neutron in W/m³
```

### Species Tracking

`SpeciesState` holds fuel and ash densities:

```python
@dataclass
class SpeciesState:
    n_D: Array    # Deuterium density [m⁻³]
    n_T: Array    # Tritium density [m⁻³]
    n_He3: Array  # Helium-3 density [m⁻³]
    n_He4: Array  # Helium-4 (ash) density [m⁻³]
```

`SpeciesTracker` computes source terms:

```python
tracker = SpeciesTracker()
dn_dt = tracker.compute_sources(species, rates)
# Returns density change rates accounting for fuel consumption and ash production
```

### Direct Conversion

`DirectConversion` computes power recovered via magnetic induction:

$$V = -N \frac{d\Psi}{dt} \eta_{coupling}$$
$$P = \frac{V^2}{4R}$$

```python
conversion = DirectConversion(
    coil_turns=100,
    coil_radius=0.6,
    circuit_resistance=0.1,
    coupling_efficiency=0.9
)

P_electric = conversion.compute_power(psi_old, psi_new, dt)
```

## Integration with BurningPlasmaModel

The burn module integrates with `BurningPlasmaModel`:

```python
from jax_frc.models import BurningPlasmaModel

model = BurningPlasmaModel(
    mhd_core=ResistiveMHD.from_config({...}),
    burn=BurnPhysics(fuels=("DT", "DD")),
    species_tracker=SpeciesTracker(),
    transport=TransportModel(...),
    conversion=DirectConversion(...)
)
```

## Parameters

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| `fuels` | Enabled reactions | `("DT",)` or `("DT", "DD")` |
| `coil_turns` | Direct conversion coil turns | 50-200 |
| `coil_radius` | Conversion coil radius | 0.4-0.8 m |
| `circuit_resistance` | Load resistance | 0.01-1.0 Ω |
| `coupling_efficiency` | Magnetic coupling factor | 0.8-0.95 |

## References

- Bosch & Hale, "Improved formulas for fusion cross-sections", Nuclear Fusion 32 (1992)
