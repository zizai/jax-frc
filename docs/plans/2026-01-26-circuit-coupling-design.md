# Circuit Coupling Design for Burning Plasma Model

**Date:** 2026-01-26
**Status:** Proposed
**Goal:** Add bidirectional plasma-circuit coupling with multi-coil energy extraction

## Overview

This design adds a circuit layer to the burning plasma model that enables:

1. **Multi-coil energy extraction** - Array of pickup coils at different axial positions, each with its own RLC circuit, capturing spatial structure of expanding plasma
2. **External driven circuits** - External coils with circuit dynamics (L, R, C) and voltage/current sources, supporting feedback control
3. **Bidirectional coupling** - Plasma flux induces EMF in circuits; external coil currents contribute to plasma B-field

Primary focus is accurate direct energy conversion modeling during plasma expansion/decay. External circuit capability supports active control scenarios.

## Architecture

```
BurningPlasmaModel
├── mhd_core: ResistiveMHD          (existing)
├── burn: BurnPhysics               (existing)
├── species_tracker: SpeciesTracker (existing)
├── transport: TransportModel       (existing)
├── circuits: CircuitSystem         (NEW - replaces DirectConversion)
│   ├── pickup: PickupCoilArray     (multi-coil extraction)
│   ├── external: ExternalCircuits  (driven coils with feedback)
│   └── flux_coupling: FluxCoupling (plasma-circuit mutual inductance)
```

The `CircuitSystem` replaces the existing `DirectConversion` module, which computed power from flux change without proper circuit dynamics.

## Circuit Model

Each circuit (pickup or external) is modeled as an RLC circuit:

```
L dI/dt + R*I + (1/C)∫I dt = V_source - dΨ_plasma/dt
```

Where:
- `L` = circuit inductance [H]
- `R` = circuit resistance [Ω]
- `C` = capacitance [F] (optional, for resonant extraction)
- `V_source` = applied voltage from driver [V]
- `dΨ_plasma/dt` = induced EMF from changing plasma flux linkage [V]

## Data Structures

### Circuit State (JAX pytree)

```python
@dataclass(frozen=True)
class CircuitState:
    # Pickup coil array (n_pickup coils)
    I_pickup: Array        # shape (n_pickup,) - coil currents [A]
    Q_pickup: Array        # shape (n_pickup,) - capacitor charge [C]

    # External circuits (n_external coils)
    I_external: Array      # shape (n_external,) - coil currents [A]
    Q_external: Array      # shape (n_external,) - capacitor charge [C]

    # Cached flux linkages (for computing dΨ/dt)
    Psi_pickup: Array      # shape (n_pickup,) - flux through pickup coils [Wb]
    Psi_external: Array    # shape (n_external,) - flux through external coils [Wb]

    # Diagnostics
    P_extracted: float     # total extracted power [W]
    P_dissipated: float    # power lost to resistance [W]
```

### Circuit Parameters (static, frozen)

```python
@dataclass(frozen=True)
class CircuitParams:
    L: Array      # inductance [H], shape (n_coils,)
    R: Array      # resistance [Ω], shape (n_coils,)
    C: Array      # capacitance [F], shape (n_coils,) - use inf for no capacitor
```

### Updated Burning Plasma State

```python
@dataclass(frozen=True)
class BurningPlasmaState:
    mhd: State
    species: SpeciesState
    rates: ReactionRates
    power: PowerSources
    circuits: CircuitState  # replaces ConversionState
```

## Pickup Coil Array

Multi-coil extraction system for capturing spatial structure:

```python
@dataclass(frozen=True)
class PickupCoilArray:
    """Array of pickup coils at different axial positions.

    Attributes:
        z_positions: Axial positions of coil centers [m], shape (n_coils,)
        radii: Coil radii [m], shape (n_coils,)
        n_turns: Turns per coil, shape (n_coils,)
        params: Circuit parameters (L, R, C) for each coil
        load_resistance: External load resistance [Ω], shape (n_coils,)
    """
    z_positions: Array
    radii: Array
    n_turns: Array
    params: CircuitParams
    load_resistance: Array

    def compute_flux_linkages(self, B: Array, geometry: Geometry) -> Array:
        """Compute Ψ = N * ∫B·dA for each coil.

        Uses vectorized integration: for each coil, sum Bz * 2πr * dr
        over cells where r < coil_radius at the coil's z-position.

        Returns:
            Psi: shape (n_coils,) - flux linkage for each coil [Wb]
        """
        ...
```

Power extraction:
```python
P_load = I_pickup**2 * load_resistance  # power to useful load
P_dissipated = I_pickup**2 * params.R   # resistive losses in coil
P_extracted = jnp.sum(P_load)           # total useful power
```

Coils can be configured for resonant extraction by setting `C` to tune the LC frequency to the plasma expansion timescale.

## External Circuits

External circuits support voltage/current sources and feedback control:

```python
@dataclass(frozen=True)
class CoilGeometry:
    """Physical geometry of external coil."""
    z_center: float   # axial position [m]
    radius: float     # coil radius [m]
    length: float     # coil length [m]
    n_turns: int      # number of turns


@dataclass(frozen=True)
class CircuitDriver:
    """Voltage or current source for external circuit.

    Attributes:
        mode: "voltage" | "current" | "feedback"
        waveform: For voltage/current mode - callable(t) -> value
        feedback_gains: For feedback mode - (Kp, Ki, Kd) PID gains
        feedback_target: For feedback mode - callable(state) -> target value
    """
    mode: str
    waveform: Optional[Callable[[float], float]] = None
    feedback_gains: Optional[tuple[float, float, float]] = None
    feedback_target: Optional[Callable] = None


@dataclass(frozen=True)
class ExternalCircuit:
    """Single external circuit (coil + driver)."""
    coil: CoilGeometry
    params: CircuitParams
    driver: CircuitDriver
```

Feedback example (maintain flux at separatrix):
```python
driver = CircuitDriver(
    mode="feedback",
    feedback_gains=(1e3, 1e2, 0.0),  # Kp, Ki, Kd
    feedback_target=lambda state: state.mhd.psi[separatrix_idx]
)
```

## Flux Linkage Computation

Coupling between plasma and circuits through mutual flux linkage:

```python
@dataclass(frozen=True)
class FluxCoupling:
    """Computes flux linkages between plasma and all circuits."""

    def plasma_to_coils(
        self, B_plasma: Array, geometry: Geometry,
        pickup: PickupCoilArray, external: ExternalCircuits
    ) -> tuple[Array, Array]:
        """Compute plasma flux threading each circuit.

        Returns:
            Psi_pickup: shape (n_pickup,)
            Psi_external: shape (n_external,)
        """
        ...

    def coils_to_plasma(
        self, I_external: Array, external: ExternalCircuits,
        geometry: Geometry
    ) -> Array:
        """Compute B-field from external coil currents.

        Returns:
            B_coils: shape (nr, nz, 3) - field to add to plasma B
        """
        ...
```

In the step function:
```python
# 1. Get current plasma flux through circuits
Psi_new = flux_coupling.plasma_to_coils(B_plasma, ...)

# 2. Compute induced EMF: V_ind = -N * dΨ/dt
dPsi_dt = (Psi_new - state.circuits.Psi_pickup) / dt
V_induced = -pickup.n_turns * dPsi_dt

# 3. This enters the circuit ODE as a source term
```

## Solver Integration

Circuit integration works with existing `TimeController` and adaptive timestep:

```python
@dataclass(frozen=True)
class CircuitSystem:
    """Complete circuit system integrated with MHD solver."""
    pickup: PickupCoilArray
    external: ExternalCircuits
    flux_coupling: FluxCoupling

    def step(
        self,
        circuit_state: CircuitState,
        B_plasma: Array,
        geometry: Geometry,
        t: float,
        dt: float,
    ) -> CircuitState:
        """Advance all circuits by dt.

        Uses same dt as MHD for synchronization.
        Circuit timescales (L/R) are typically faster than MHD,
        so we subcycle if needed using lax.scan.
        """
        ...
```

Subcycling (JIT-compatible):
```python
def _subcycle_step(self, state, B_plasma, geometry, t, dt):
    # Estimate circuit timescale: tau = L/R
    tau_min = jnp.min(self.pickup.params.L / self.pickup.params.R)

    # Number of substeps (clamped for JIT stability)
    n_sub = jnp.clip(jnp.ceil(dt / (0.1 * tau_min)), 1, MAX_SUBSTEPS)
    dt_sub = dt / n_sub

    # lax.scan over substeps
    def body(carry, _):
        return self._circuit_ode_step(carry, dt_sub), None

    final_state, _ = lax.scan(body, state, None, length=n_sub)
    return final_state
```

The `BurningPlasmaModel.step()` calls `circuits.step()` after the MHD update, using the new B-field to compute induced EMFs.

## Configuration

Users specify circuits through the config dictionary:

```python
config = {
    "fuels": ["DT"],
    "mhd": {"resistivity": {"type": "spitzer"}},
    "transport": {"D_particle": 0.1, "chi_e": 1.0},

    "circuits": {
        "pickup_array": {
            "z_positions": [-0.5, -0.25, 0.0, 0.25, 0.5],
            "radii": [0.6, 0.6, 0.6, 0.6, 0.6],
            "n_turns": [100, 100, 100, 100, 100],
            "L": [1e-3, 1e-3, 1e-3, 1e-3, 1e-3],
            "R": [0.1, 0.1, 0.1, 0.1, 0.1],
            "C": null,
            "load_resistance": [1.0, 1.0, 1.0, 1.0, 1.0]
        },

        "external": [
            {
                "name": "compression_coil",
                "z_center": 0.0,
                "radius": 0.8,
                "length": 1.0,
                "n_turns": 50,
                "L": 5e-3,
                "R": 0.05,
                "C": null,
                "driver": {
                    "mode": "voltage",
                    "waveform": {"type": "ramp", "V0": 0, "V1": 1000, "t_ramp": 1e-4}
                }
            }
        ]
    }
}

model = BurningPlasmaModel.from_config(config)
```

Supported waveform types:
- `ramp`: Linear ramp from V0 to V1 over t_ramp
- `sinusoid`: `A * sin(2π*f*t + φ)`
- `pulse`: Square pulse with rise/fall times
- `crowbar`: Step down to zero at specified time (for formation circuits)

## File Structure

**New files:**
```
jax_frc/
├── circuits/
│   ├── __init__.py          # exports CircuitSystem, CircuitState
│   ├── state.py             # CircuitState, CircuitParams dataclasses
│   ├── pickup.py            # PickupCoilArray
│   ├── external.py          # ExternalCircuits, CircuitDriver, CoilGeometry
│   ├── coupling.py          # FluxCoupling
│   ├── system.py            # CircuitSystem (orchestrates all)
│   └── waveforms.py         # Predefined waveform functions
tests/
├── test_circuits.py         # Unit tests for circuit components
├── test_circuit_coupling.py # Integration tests with plasma
```

**Files to modify:**
- `jax_frc/models/burning_plasma.py` - Replace `DirectConversion` with `CircuitSystem`
- `jax_frc/burn/__init__.py` - Remove `ConversionState` export (moved to circuits)

## Testing Strategy

### Unit Tests
- Circuit ODE integration accuracy (compare to analytical RLC solutions)
- Flux linkage computation (compare to known geometries)
- Waveform generation

### Physics Invariants
1. **Energy conservation**: `P_extracted + P_dissipated = -d(E_magnetic)/dt`
2. **Flux conservation**: In superconducting limit (R→0), total flux through circuit is constant
3. **Backward compatibility**: Single pickup coil with R-only should reproduce existing `DirectConversion` behavior

### Integration Tests
- Multi-coil extraction from expanding FRC
- External coil driving plasma compression
- Feedback control maintaining equilibrium

## Implementation Notes

### JIT Compatibility
- All arrays use JAX arrays, no Python lists in hot path
- Subcycling uses `lax.scan`, not Python for-loops
- Waveforms compiled to JAX functions at construction time
- Feedback targets must be JIT-traceable

### Performance
- Flux integration vectorized over all coils simultaneously
- Circuit ODE uses implicit midpoint for stability with stiff L/R ratios
- External coil B-field uses existing `Solenoid.B_field` infrastructure

### Numerical Stability
- Subcycling ensures circuit timestep resolves L/R timescale
- Capacitor charge Q tracked separately to avoid differentiating current
- Flux linkage cached to compute clean dΨ/dt
