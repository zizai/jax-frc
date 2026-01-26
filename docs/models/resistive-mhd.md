# Resistive MHD

Single-fluid resistive MHD model using flux function formulation. Best for engineering design, circuit optimization, and formation dynamics.

## Physics Overview

Resistive MHD treats the plasma as a single conducting fluid with finite resistivity. Key assumptions:
- **Single fluid**: Ions and electrons move together (no Hall effect)
- **MHD ordering**: Slow timescales compared to cyclotron frequency
- **Collisional**: Resistivity from Coulomb collisions or anomalous effects

**When to use**: Circuit design, coil optimization, formation dynamics where two-fluid effects are negligible.

**When NOT to use**: FRC stability analysis (fails on tilt mode), kinetic effects, fast ion physics.

## Governing Equations

### Flux Function Evolution

$$\frac{\partial \psi}{\partial t} + \mathbf{v} \cdot \nabla \psi = \frac{\eta}{\mu_0} \Delta^* \psi$$

Where:
- $\psi$ is the poloidal flux function
- $\eta$ is the plasma resistivity
- $\Delta^*$ is the Grad-Shafranov operator: $\Delta^* = r \frac{\partial}{\partial r}\left(\frac{1}{r}\frac{\partial}{\partial r}\right) + \frac{\partial^2}{\partial z^2}$

### Current Density

The toroidal current density is:

$$J_\phi = -\frac{\Delta^* \psi}{\mu_0 r}$$

### Magnetic Field from Flux

In cylindrical coordinates:
- $B_r = -\frac{1}{r}\frac{\partial \psi}{\partial z}$
- $B_z = \frac{1}{r}\frac{\partial \psi}{\partial r}$

## Implementation

### Class: `ResistiveMHD`

Location: `jax_frc/models/resistive_mhd.py`

```python
@dataclass(frozen=True)
class ResistiveMHD(PhysicsModel):
    resistivity: ResistivityModel
    external_field: Optional[CoilField] = None
```

### Method Mapping

| Physics | Method | Description |
|---------|--------|-------------|
| $\Delta^* \psi$ | `_laplace_star()` | Grad-Shafranov operator |
| $J_\phi$ | Computed in `compute_rhs()` | From $-\Delta^* \psi / (\mu_0 r)$ |
| $\eta(J)$ | `resistivity.compute()` | Resistivity model evaluation |
| Total B field | `get_total_B()` | Includes external coil field if present |

### Resistivity Models

Two options available:
- **Spitzer**: Classical collisional resistivity $\eta \propto T^{-3/2}$
- **Chodura**: Anomalous resistivity that activates at high current density

## Parameters

| Parameter | Physical Meaning | Typical Range | Default |
|-----------|------------------|---------------|---------|
| `eta_0` | Base resistivity | 1e-8 to 1e-4 Ω·m | 1e-6 |
| `eta_anom` | Anomalous resistivity (Chodura) | 1e-5 to 1e-2 Ω·m | 1e-3 |
| `j_crit` | Critical current for anomalous onset | 1e5 to 1e7 A/m² | 1e6 |

### Tuning Guidance

- **Higher `eta_0`**: Faster magnetic diffusion, shorter resistive timescale
- **Higher `eta_anom`**: Faster reconnection at current sheets
- **Lower `j_crit`**: Earlier onset of anomalous effects

## Solver Compatibility

| Solver | Compatible | Notes |
|--------|------------|-------|
| `RK4Solver` | ✓ Recommended | Standard explicit integration |
| `EulerSolver` | ✓ | First-order, for testing only |
| `ImexSolver` | ✓ | For stiff resistivity profiles |
| `SemiImplicitSolver` | ✗ | Designed for Extended MHD |

## Validation

### Tests

- `tests/test_resistive_mhd.py`: Unit tests for operators and RHS
- `tests/invariants/conservation.py`: Flux conservation checks

### Known Limitations

1. **Predicts FRC instability**: Single-fluid MHD misses stabilizing FLR effects
2. **No temperature evolution**: Isothermal assumption
3. **Axisymmetric only**: 2D $(r, z)$ geometry

### Literature Comparison

Model based on Lamy Ridge formulation. For FRC-specific validation, see [Belova et al. comparison](../modules/comparisons.md).

## Example Usage

```python
from jax_frc import Geometry, ResistiveMHD, RK4Solver, Simulation, TimeController
import jax.numpy as jnp

# Setup geometry
geometry = Geometry(
    coord_system='cylindrical',
    nr=64, nz=128,
    r_min=0.01, r_max=1.0,
    z_min=-1.0, z_max=1.0
)

# Create model with Chodura resistivity
model = ResistiveMHD.from_config({
    'resistivity': {
        'type': 'chodura',
        'eta_0': 1e-6,
        'eta_anom': 1e-3,
        'j_crit': 1e6
    }
})

# Setup solver and time control
solver = RK4Solver()
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

# Create and run simulation
sim = Simulation(
    geometry=geometry,
    model=model,
    solver=solver,
    time_controller=time_controller
)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final_state = sim.run_steps(500)
```
