# Scenario Framework Design for FRC Experiments

**Date:** 2026-01-26
**Status:** Draft
**Authors:** Design session with Claude

---

## Overview

This design adds a composable phase pipeline framework to support full FRC experiment cycles: formation, translation/acceleration, merging, compression, and nuclear burn. The framework enables both individual phase studies and complete Helion-style operational sequences.

## Goals

- Support multiple experiment types through composable phases
- Enable MHD and Hybrid model comparisons for each phase
- Validate against Belova et al. (arXiv:2501.03425v1) merging results
- Model D-D, D-T, and D-He3 fusion burn with appropriate product handling

## Non-Goals

- 3D simulations (remains 2D axisymmetric)
- Real-time control system integration
- Multi-device/distributed computing (single GPU focus)

---

## Architecture

### Module Structure

```
jax_frc/
  scenarios/
    __init__.py
    phase.py              # Phase base class and lifecycle
    scenario.py           # Scenario runner and orchestration
    transitions.py        # Condition and time-based triggers

    phases/
      __init__.py
      formation.py        # Theta-pinch / flux coil FRC formation
      translation.py      # Acceleration through device
      merging.py          # Two-FRC collision and reconnection
      compression.py      # Adiabatic compression
      burn.py             # Nuclear burn phase

  boundaries/
    time_dependent.py     # Time-varying mirror field BCs

  diagnostics/
    merging.py            # Merging-specific metrics

  burn/
    __init__.py
    reactions.py          # Reaction rate coefficients
    products.py           # Energy/momentum deposition
    fuel.py               # Fuel tracking and depletion
    fast_ions.py          # Kinetic fast ion species (hybrid)

examples/
  merging_examples.py     # Two-FRC setups, paper validation cases
```

---

## Phase Framework

### Phase Base Class

```python
@dataclass
class Phase:
    name: str
    transition: Transition

    def setup(self, state: State, geometry: Geometry, config: dict) -> State:
        """Called once when phase begins. Modify state, set BCs, etc."""
        return state

    def step_hook(self, state: State, t: float) -> State:
        """Called each timestep. For time-varying BCs like compression."""
        return state

    def is_complete(self, state: State, t: float) -> tuple[bool, str]:
        """Check if phase should end. Delegates to transition."""
        return self.transition.evaluate(state, t)

    def on_complete(self, state: State) -> State:
        """Called once when phase ends. Cleanup, prepare for next phase."""
        return state
```

### Transition Logic

Hybrid approach: condition-based triggers with time-based fallback.

```python
@dataclass
class Transition:
    condition: Callable[[State, float], bool] | None  # Physics-based
    timeout: float | None                              # Time-based fallback

    def evaluate(self, state: State, t: float) -> tuple[bool, str]:
        if self.condition and self.condition(state, t):
            return True, "condition_met"
        if self.timeout and t >= self.timeout:
            return True, "timeout"
        return False, ""
```

### Built-in Transition Conditions

- `separation_below(threshold)` - For merging completion
- `temperature_above(threshold)` - For burn ignition
- `flux_below(threshold)` - For confinement loss
- `velocity_below(threshold)` - For translation completion

Composable: `any_of(cond1, cond2)`, `all_of(cond1, cond2)`

---

## Phase Implementations

### FormationPhase

- **setup**: Initialize uniform plasma with embedded flux coil fields
- **step_hook**: Ramp coil currents to compress plasma into FRC
- **Transition**: Closed flux surfaces detected (psi has local maximum away from axis)
- **Output**: Single FRC equilibrium

### TranslationPhase

- **setup**: Apply mirror field gradient or moving coil profile
- **step_hook**: Update time-dependent acceleration fields
- **Transition**: FRC center reaches target z-position, or velocity reaches target
- **Output**: FRC with axial momentum

### MergingPhase

- **setup**: Mirror-flip state to create two-FRC configuration, apply initial velocities
- **step_hook**: Optionally ramp compression mirror fields at boundaries
- **Transition**: Separation dZ below threshold (complete merge) or stabilizes (doublet)
- **Output**: Merged FRC (single or doublet)

### CompressionPhase

- **setup**: Configure radial compression field profile
- **step_hook**: Ramp external field strength at boundary (Bext(t))
- **Transition**: Target field strength reached or pressure/temperature target hit
- **Output**: Compressed, heated FRC

### BurnPhase

- **setup**: Enable nuclear reaction source terms in physics model
- **step_hook**: Track fusion power, energy deposition, fuel depletion
- **Transition**: Fuel depleted, confinement lost, or target burn duration reached
- **Output**: Post-burn state with fusion yield diagnostics

---

## Two-FRC Initial Conditions

Mirror-flip procedure following Belova et al.:

1. Take single-FRC equilibrium with separatrix at z in [-Zs, +Zs]
2. Place domain with total length 2*Zc, midplane at z=0
3. Position FRC1 at z = -dZ/2, FRC2 at z = +dZ/2 (centers at magnetic nulls)
4. FRC2 is axially mirrored: psi2(r,z) = psi1(r, -z)
5. Velocities are antisymmetric: Vz1 = +V0, Vz2 = -V0

### Midplane Singularity Handling

Direct superposition can create current singularities at z=0. Solution:
- After combining fields, re-solve Grad-Shafranov with RHS = R*Jphi from combined current
- This smooths the midplane region while preserving FRC structure

---

## Time-Dependent Boundary Conditions

### TimeDependentMirrorBC

```python
@dataclass
class TimeDependentMirrorBC(BoundaryCondition):
    """Mirror field with time-varying strength at z boundaries."""

    base_field: float           # B0 at t=0
    mirror_ratio_final: float   # B_end/B0 (e.g., 1.5)
    ramp_time: float            # T in Alfven times
    profile: str                # "cosine" or "linear"

    def apply(self, state: State, geometry: Geometry, t: float) -> State:
        # Compute current mirror ratio from t
        # Update Aphi at z = +/-Zc boundary
```

### Compression Profile (from paper)

- Spatial: dAphi(z) ~ 0.5(1 - cos(pi*z/Zc)) - strongest at ends, zero at midplane
- Temporal: f(t) ~ (1 - cos(pi*t/T)) - smooth ramp over time T

---

## Merging Diagnostics

```python
@dataclass
class MergingDiagnostics(DiagnosticProbe):
    """Compute merging metrics each output step."""

    def compute(self, state: State, geometry: Geometry) -> dict:
        return {
            "separation_dz": self._find_null_separation(state),
            "separatrix_radius": self._find_rs(state),
            "elongation": self._find_elongation(state),
            "separatrix_beta": self._compute_beta_s(state),
            "peak_pressure": jnp.max(state.p),
            "null_positions": self._find_null_positions(state),
            "axial_velocity_at_null": self._vz_at_null(state),
            "reconnection_rate": self._compute_reconnection_rate(state),
        }
```

### Key Algorithms

- **Null finding**: Locate points where Bz=0 and Br=0 (psi has saddle/extremum)
- **Separatrix tracing**: Contour of psi passing through X-point
- **Reconnection rate**: dpsi/dt at X-point, or electric field Ephi at reconnection site
- **Separation dZ**: Distance between magnetic nulls along axis

---

## Nuclear Burn Physics

### Reaction Database

```python
@dataclass
class Reaction:
    name: str                           # e.g., "D-He3"
    reactants: tuple[str, str]          # ("D", "He3")
    products: tuple[str, ...]           # ("He4", "p")
    Q_value: float                      # MeV released
    sigma_v: Callable[[float], float]   # <sigma*v>(T) in m^3/s
    branching_ratio: float = 1.0
```

### Supported Reactions

| Reaction | Q (MeV) | Products |
|----------|---------|----------|
| D + D -> He3 + n | 3.27 | He3 (0.82 MeV), n (2.45 MeV) |
| D + D -> T + p | 4.03 | T (1.01 MeV), p (3.02 MeV) |
| D + T -> He4 + n | 17.6 | He4 (3.5 MeV), n (14.1 MeV) |
| D + He3 -> He4 + p | 18.3 | He4 (3.6 MeV), p (14.7 MeV) |

### Energy Deposition (MHD Mode)

- Charged products (alpha, p, T, He3): deposit to ions locally
- Neutrons: escape (contribute to yield, not plasma heating)
- Split based on slowing-down physics: ~80% to ions at FRC temperatures

### Fast Ion Tracking (Hybrid Mode)

```python
@dataclass
class FastIonSpecies:
    name: str              # "alpha", "proton", "triton"
    mass: float            # In proton masses
    charge: float          # In elementary charges
    birth_energy: float    # MeV
    particles: Particles   # Reuse existing particle data structure
```

**Source term (birth):** Each timestep, compute local fusion rate, probabilistically spawn particles with isotropic velocity at birth energy.

**Sink term (thermalization):** When fast ion energy drops below ~3x thermal energy, transfer to thermal ion population.

---

## Scenario Runner

```python
@dataclass
class Scenario:
    name: str
    phases: list[Phase]
    physics_model: PhysicsModel
    geometry: Geometry
    initial_state: State | None

    def run(self) -> ScenarioResult:
        state = self.initial_state
        results = []

        for phase in self.phases:
            phase_result = self._run_phase(phase, state)
            results.append(phase_result)
            state = phase_result.final_state

            if phase_result.termination == "timeout":
                log.warning(f"{phase.name} ended by timeout")

        return ScenarioResult(phases=results)
```

### Configuration via YAML

```yaml
scenario: helion_cycle
physics: hybrid_kinetic
fuel: D-He3
phases:
  merging:
    separation: 180
    initial_velocity: 0.2
    compression:
      mirror_ratio: 1.5
      ramp_time: 19
  burn:
    timeout: 50
```

---

## Validation Plan

### Test Cases from Belova et al.

| Case | Parameters | Expected Outcome |
|------|------------|------------------|
| 1 | S*=25.6, E=2.9, xs=0.69, no compression | Partial merge, doublet (dZ~40) |
| 2 | S*=20, E=1.5, xs=0.53, no compression | Complete merge by ~5-7 tA |
| 4 | Case 1 params + compression (1.5x mirror) | Complete merge by ~20-25 tA |

### Success Criteria

- MHD merging dynamics match paper figures (dZ vs t curves)
- Hybrid shows Hall reconnection signatures (quadrupole Bphi)
- Burn phase produces expected fusion rates for given n, T
- Full cycle runs without phase transition failures

---

## Implementation Order

1. **Phase framework** - Base classes, transitions, scenario runner
2. **MergingPhase** - Two-FRC setup, diagnostics, validate against paper
3. **Time-dependent BCs** - Compression mirror fields
4. **CompressionPhase** - Adiabatic heating
5. **Burn module** - Reactions, MHD energy deposition
6. **Fast ions** - Hybrid mode product tracking
7. **FormationPhase** - FRC creation from scratch
8. **TranslationPhase** - Acceleration physics
9. **Full cycle integration** - End-to-end Helion scenario
