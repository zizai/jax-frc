# Extended MHD

Two-fluid MHD model with Hall effect and electron pressure. Best for global stability analysis, translation dynamics, and transport scaling.

## Physics Overview

Extended MHD captures two-fluid effects by including the Hall term in Ohm's law. Key features:
- **Hall effect**: Decouples ion and electron motion at scales below ion skin depth
- **Whistler waves**: High-frequency waves requiring implicit treatment
- **Electron pressure**: Separate electron pressure gradient term

**When to use**: FRC stability analysis, translation dynamics, scenarios where Hall physics matters.

**When NOT to use**: Kinetic beam effects, scenarios requiring particle distributions.

## Governing Equations

### Extended Ohm's Law

$$\mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J} + \frac{\mathbf{J} \times \mathbf{B}}{ne} - \frac{\nabla p_e}{ne}$$

Where the additional terms beyond resistive MHD are:
- $\frac{\mathbf{J} \times \mathbf{B}}{ne}$: Hall term (two-fluid effect)
- $\frac{\nabla p_e}{ne}$: Electron pressure gradient

### Temperature Evolution

$$\frac{3}{2} n \frac{\partial T}{\partial t} = -\nabla \cdot \mathbf{q} + \eta J^2 - p \nabla \cdot \mathbf{v}$$

Where $\mathbf{q}$ includes both classical and anomalous heat flux.

### Characteristic Scales

- **Ion skin depth**: $d_i = c/\omega_{pi}$ (Hall effect important below this scale)
- **Whistler frequency**: $\omega_w \sim k^2 d_i^2 \Omega_i$ (requires implicit treatment)

## Implementation

### Class: `ExtendedMHD`

Location: `jax_frc/models/extended_mhd.py`

```python
@dataclass(frozen=True)
class ExtendedMHD(PhysicsModel):
    resistivity: ResistivityModel
    hall_enabled: bool = True
    electron_pressure: bool = True
    halo_density: HaloDensityModel = None
```

### Method Mapping

| Physics | Method | Description |
|---------|--------|-------------|
| Extended Ohm's law | `_extended_ohm_law()` | Full E-field with Hall term |
| Hall term | Computed in `_extended_ohm_law()` | $(J \times B)/(ne)$ |
| Temperature RHS | `_compute_temperature_rhs()` | Heat equation |
| Current density | `_compute_current()` | $\nabla \times B / \mu_0$ |
| Curl of E | `_compute_curl_E()` | For Faraday's law |

### Halo Density Model

Handles vacuum regions to prevent division-by-zero in Hall term:

```python
class HaloDensityModel:
    def apply(self, n: Array) -> Array:
        """Enforce minimum density in vacuum regions."""
        return jnp.maximum(n, self.n_halo)
```

## Parameters

| Parameter | Physical Meaning | Typical Range | Default |
|-----------|------------------|---------------|---------|
| `eta_0` | Base resistivity | 1e-6 to 1e-3 Ω·m | 1e-4 |
| `hall_enabled` | Include Hall term | True/False | True |
| `electron_pressure` | Include ∇p_e term | True/False | True |
| `n_halo` | Minimum halo density | 1e16 to 1e18 m⁻³ | 1e17 |
| `T_boundary` | Temperature BC value | 1 to 100 eV | 10 |

### Tuning Guidance

- **`hall_enabled=False`**: Reduces to resistive MHD (faster, less accurate)
- **Higher `n_halo`**: More stable numerics, less accurate vacuum
- **Lower `T_boundary`**: Colder edge, affects thermal confinement

## Solver Compatibility

| Solver | Compatible | Notes |
|--------|------------|-------|
| `SemiImplicitSolver` | ✓ Recommended | Handles Whistler waves implicitly |
| `ImexSolver` | ✓ | Alternative implicit-explicit |
| `RK4Solver` | ✗ | Whistler CFL too restrictive |
| `EulerSolver` | ✗ | Unstable for Hall physics |

## Validation

### Tests

- `tests/test_extended_mhd.py`: Unit tests for Hall term, temperature evolution
- `tests/invariants/conservation.py`: Energy conservation with Hall term

### Known Limitations

1. **Misses FLR effects**: Finite Larmor radius stabilization not captured
2. **No beam physics**: Fast ion effects require hybrid model
3. **Computationally expensive**: Semi-implicit solve each timestep

### Comparison to Codes

Based on NIMROD formulation. Hall term implementation validated against analytical dispersion relations.

## Example Usage

```python
from jax_frc import Geometry, ExtendedMHD, SemiImplicitSolver, Simulation, TimeController
import jax.numpy as jnp

# Setup geometry
geometry = Geometry(
    coord_system='cylindrical',
    nr=32, nz=64,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

# Create model with Hall effect
model = ExtendedMHD.from_config({
    'resistivity': {'type': 'spitzer', 'eta_0': 1e-4},
    'hall': {'enabled': True},
    'halo': {'n_halo': 1e17}
})

# Semi-implicit solver required for Whistler waves
solver = SemiImplicitSolver()
time_controller = TimeController(cfl_safety=0.1, dt_max=1e-6)

# Create and run simulation
sim = Simulation(
    geometry=geometry,
    model=model,
    solver=solver,
    time_controller=time_controller
)
sim.initialize(psi_init=lambda r, z: jnp.exp(-r**2 - z**2))
final_state = sim.run_steps(100)
```
