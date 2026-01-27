# Magnetic Diffusion Validation Cases Design

**Date:** 2026-01-27
**Status:** Approved
**Goal:** Replace heat diffusion test with magnetic diffusion validation cases for two magnetic Reynolds number regimes

## Overview

Create two validation configurations testing magnetic diffusion physics:

1. **MagneticDiffusionConfiguration** (Rm ≪ 1): Resistive diffusion dominates
2. **FrozenFluxConfiguration** (Rm ≫ 1): Ideal MHD, flux frozen to plasma

Both configurations support multiple physics models: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic.

## Physics Background

The magnetic induction equation:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}$$

where η = 1/(μ₀σ) is the magnetic diffusivity.

**Magnetic Reynolds Number**: Rm = vL/η

| Rm | Regime | Dominant Term | Equation |
|----|--------|---------------|----------|
| ≪ 1 | Diffusion | Resistive | ∂B/∂t ≈ η∇²B |
| ≫ 1 | Frozen-in | Advective | ∂B/∂t ≈ ∇×(v×B) |

## File Structure

```
jax_frc/configurations/
├── __init__.py                    # Update exports
├── magnetic_diffusion.py          # NEW: MagneticDiffusionConfiguration
├── frozen_flux.py                 # NEW: FrozenFluxConfiguration
├── base.py                        # Keep
├── validation_benchmarks.py       # Keep
└── (delete analytic.py)
```

## MagneticDiffusionConfiguration (Rm ≪ 1)

**File**: `jax_frc/configurations/magnetic_diffusion.py`

**Physics**: Pure resistive diffusion with v=0

**Geometry**: 1D in z (axial), Gaussian B_z profile

**Analytic Solution**:
$$B_z(z,t) = B_{peak} \sqrt{\frac{\sigma_0^2}{\sigma_0^2 + 2\eta t}} \exp\left(-\frac{z^2}{2(\sigma_0^2 + 2\eta t)}\right)$$

```python
@dataclass
class MagneticDiffusionConfiguration(AbstractConfiguration):
    """1D magnetic diffusion test (Rm << 1).

    Physics: ∂B/∂t = η∇²B (resistive diffusion, no flow)

    Initial condition: Gaussian B_z profile along z
    Analytic solution: Spreading Gaussian

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "magnetic_diffusion"
    description: str = "1D magnetic field diffusion (Rm << 1)"

    # Grid
    nr: int = 8
    nz: int = 128
    z_extent: float = 2.0  # Domain: z ∈ [-z_extent, z_extent]

    # Physics
    B_peak: float = 1.0          # Peak B_z [T]
    sigma: float = 0.3           # Initial Gaussian width [m]
    eta: float = 1e-2            # Magnetic diffusivity [m²/s]

    # Model selection
    model_type: str = "resistive_mhd"  # resistive_mhd | extended_mhd | plasma_neutral | hybrid_kinetic

    def build_geometry(self) -> Geometry:
        """Cylindrical geometry, minimal r resolution (1D in z)."""
        ...

    def build_initial_state(self, geometry: Geometry) -> State:
        """Gaussian B_z profile, v=0."""
        ...

    def build_model(self):
        """Build physics model with appropriate resistivity."""
        if self.model_type == "resistive_mhd":
            return ResistiveMHD(resistivity=ConstantResistivity(eta_0=self.eta))
        elif self.model_type == "extended_mhd":
            return ExtendedMHD(resistivity=..., thermal=None)
        elif self.model_type == "plasma_neutral":
            return PlasmaNeutralModel(resistivity=...)
        elif self.model_type == "hybrid_kinetic":
            return HybridKineticModel(resistivity=...)

    def analytic_solution(self, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute analytic B_z at time t."""
        sigma_eff_sq = self.sigma**2 + 2 * self.eta * t
        amplitude = self.B_peak * jnp.sqrt(self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-z**2 / (2 * sigma_eff_sq))

    def diffusion_timescale(self) -> float:
        """Characteristic diffusion time τ = σ²/(2η)."""
        return self.sigma**2 / (2 * self.eta)
```

## FrozenFluxConfiguration (Rm ≫ 1)

**File**: `jax_frc/configurations/frozen_flux.py`

**Physics**: Ideal MHD advection with negligible resistivity

**Geometry**: Radial (annular), B_φ with uniform radial expansion v_r = v₀

**Analytic Solution** (flux conservation):
$$B_\phi(r,t) = B_0 \cdot \frac{r_0}{r_0 + v_r t}$$

```python
@dataclass
class FrozenFluxConfiguration(AbstractConfiguration):
    """Frozen-in flux test (Rm >> 1).

    Physics: ∂B/∂t ≈ ∇×(v×B) (ideal MHD, flux frozen to plasma)

    Setup: Radial geometry with uniform expansion v_r = v₀
    Initial: Uniform B_φ in annular region
    Analytic: B_φ(t) = B₀ · r₀/(r₀ + v₀t) from flux conservation

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "frozen_flux"
    description: str = "Frozen-in magnetic flux advection (Rm >> 1)"

    # Grid (annular to avoid axis singularity)
    nr: int = 64
    nz: int = 8
    r_min: float = 0.2
    r_max: float = 1.0

    # Physics
    B_phi_0: float = 1.0         # Initial B_φ [T]
    v_r: float = 0.1             # Radial expansion velocity [m/s]
    eta: float = 1e-8            # Very small resistivity (Rm >> 1)

    # Model selection
    model_type: str = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        """Annular cylindrical geometry, high r resolution."""
        ...

    def build_initial_state(self, geometry: Geometry) -> State:
        """Uniform B_φ with prescribed radial velocity v_r."""
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 1].set(self.B_phi_0)  # B_φ component

        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 0].set(self.v_r)      # Uniform v_r

        return State(B=B, v=v, ...)

    def build_model(self):
        """Build physics model with minimal resistivity."""
        # Same structure as MagneticDiffusionConfiguration
        # but with very small eta for Rm >> 1
        ...

    def analytic_solution(self, r: jnp.ndarray, t: float) -> jnp.ndarray:
        """B_φ from flux conservation in expanding cylinder."""
        return self.B_phi_0 * self.r_min / (self.r_min + self.v_r * t)

    def magnetic_reynolds_number(self) -> float:
        """Rm = v·L/η >> 1 for this configuration."""
        L = self.r_max - self.r_min
        return self.v_r * L / self.eta
```

## Module Updates

### `__init__.py` changes

```python
# Remove
from .analytic import SlabDiffusionConfiguration

# Add
from .magnetic_diffusion import MagneticDiffusionConfiguration
from .frozen_flux import FrozenFluxConfiguration

# Update __all__
__all__ = [
    # Remove: 'SlabDiffusionConfiguration',
    # Add:
    'MagneticDiffusionConfiguration',
    'FrozenFluxConfiguration',
    # Keep existing...
]
```

### CONFIGURATION_REGISTRY updates

```python
CONFIGURATION_REGISTRY = {
    # Remove: 'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
    # Add:
    'MagneticDiffusionConfiguration': MagneticDiffusionConfiguration,
    'FrozenFluxConfiguration': FrozenFluxConfiguration,
    # Keep existing...
}
```

## Validation YAML Updates

Update or replace `validation/cases/analytic/diffusion_slab.yaml`:

```yaml
# validation/cases/analytic/magnetic_diffusion.yaml
name: magnetic_diffusion
description: "1D magnetic field diffusion (Rm << 1)"

configuration:
  class: MagneticDiffusionConfiguration
  overrides:
    model_type: resistive_mhd
    eta: 1.0e-2

acceptance:
  quantitative:
    - metric: l2_error
      field: B_z
      threshold: 0.05
```

```yaml
# validation/cases/analytic/frozen_flux.yaml
name: frozen_flux
description: "Frozen-in flux advection (Rm >> 1)"

configuration:
  class: FrozenFluxConfiguration
  overrides:
    model_type: resistive_mhd
    eta: 1.0e-8

acceptance:
  quantitative:
    - metric: l2_error
      field: B_phi
      threshold: 0.05
    - metric: flux_conservation
      threshold: 0.01
```

## Files to Delete

- `jax_frc/configurations/analytic.py`
- `validation/cases/analytic/diffusion_slab.yaml` (replace with new files)

## Testing Strategy

1. **Unit tests**: Each configuration builds valid geometry, state, model
2. **Analytic comparison**: Run short simulation, compare to analytic solution
3. **Regime verification**: Confirm Rm << 1 and Rm >> 1 as expected
4. **Model coverage**: Test each model_type option works

## Benefits

1. **Physically meaningful**: Tests actual MHD physics (magnetic diffusion) not just heat equation
2. **Two regimes**: Validates both diffusion and advection numerics
3. **Model-agnostic**: Same test case works across all physics models
4. **Different geometries**: Tests both axial (z) and radial (r) operators
