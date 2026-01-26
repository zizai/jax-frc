# Transport Module

Anomalous transport models for particle and energy diffusion.

## Overview

The `jax_frc/transport/` module provides diffusive transport beyond classical (collisional) transport. This captures turbulence-driven transport that dominates in many plasma regimes.

## Physics

### Anomalous Transport

Classical transport (Coulomb collisions) is often much smaller than observed transport in tokamaks and FRCs. Anomalous transport models this empirically:

**Particle flux**:
$$\Gamma = -D \nabla n + n \mathbf{v}_{pinch}$$

**Energy flux**:
$$\mathbf{q} = -n \chi \nabla T$$

Where:
- $D$ is the particle diffusion coefficient
- $\chi_e$, $\chi_i$ are electron and ion thermal diffusivities
- $\mathbf{v}_{pinch}$ is an optional inward pinch velocity

### Scaling

Transport coefficients may depend on local plasma parameters:
- Bohm-like: $D \sim T/B$
- Gyro-Bohm: $D \sim \rho_i T/B$
- Constant: Fixed coefficients (simplest)

## Implementation

### Class: `TransportModel`

Location: `jax_frc/transport/anomalous.py`

```python
@dataclass(frozen=True)
class TransportModel:
    D_particle: float    # Particle diffusivity [m²/s]
    chi_e: float         # Electron thermal diffusivity [m²/s]
    chi_i: float         # Ion thermal diffusivity [m²/s]
    v_pinch: float = 0.0 # Inward pinch velocity [m/s]
```

### Methods

**`particle_flux(n, geometry)`** - Compute $\Gamma$

```python
transport = TransportModel(D_particle=1.0, chi_e=5.0, chi_i=2.0)
Gamma_r, Gamma_z = transport.particle_flux(n, geometry)
```

**`energy_flux(n, T, geometry)`** - Compute $\mathbf{q}$

```python
q_r, q_z = transport.energy_flux(n, T, geometry)
```

**`flux_divergence(flux_r, flux_z, geometry)`** - Compute $\nabla \cdot \Gamma$

```python
div_Gamma = transport.flux_divergence(Gamma_r, Gamma_z, geometry)
```

### Gradient Computation

Gradients computed with central differences in cylindrical coordinates:

```python
def _gradient_r(self, f: Array, geometry: Geometry) -> Array:
    """Radial gradient: df/dr"""
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * geometry.dr)
```

## Parameters

| Parameter | Physical Meaning | Typical Range | Units |
|-----------|------------------|---------------|-------|
| `D_particle` | Particle diffusivity | 0.1 - 10 | m²/s |
| `chi_e` | Electron thermal diffusivity | 1 - 50 | m²/s |
| `chi_i` | Ion thermal diffusivity | 0.5 - 20 | m²/s |
| `v_pinch` | Inward pinch velocity | 0 - 100 | m/s |

### Tuning Guidance

- **Higher D**: Faster particle loss, shorter particle confinement time
- **Higher χ**: Faster energy loss, lower temperatures
- **χ_e > χ_i**: Typical for most plasmas (electron transport faster)
- **Non-zero v_pinch**: Can improve core density peaking

## When to Use

Enable anomalous transport when:
- Running long confinement time simulations
- Studying steady-state profiles
- Transport timescales matter (not just MHD dynamics)

Disable (or use small values) when:
- Fast MHD events (reconnection, instabilities)
- Initial formation dynamics
- Testing MHD physics in isolation

## Integration

Transport integrates with `BurningPlasmaModel`:

```python
from jax_frc.models import BurningPlasmaModel
from jax_frc.transport import TransportModel

model = BurningPlasmaModel(
    mhd_core=...,
    transport=TransportModel(
        D_particle=1.0,
        chi_e=5.0,
        chi_i=2.0
    ),
    ...
)
```

Or standalone for testing:

```python
transport = TransportModel(D_particle=1.0, chi_e=5.0, chi_i=2.0)

# Compute particle flux divergence (loss term)
Gamma_r, Gamma_z = transport.particle_flux(n, geometry)
dn_dt_transport = -transport.flux_divergence(Gamma_r, Gamma_z, geometry)
```
