# Documentation Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Comprehensive documentation update for jax-frc targeting both researchers and developers.

**Architecture:** Model-centric docs with integrated physics + implementation. New developer guide section. New module docs for burn/transport/comparisons.

**Tech Stack:** Markdown, MkDocs-compatible structure

---

## Task 1: Create Developer Architecture Doc

**Files:**
- Create: `docs/developer/architecture.md`

**Step 1: Create developer directory**

```bash
mkdir -p docs/developer
```

**Step 2: Write architecture.md**

Write the following content to `docs/developer/architecture.md`:

```markdown
# Architecture Overview

This document explains the jax-frc system architecture for developers who want to understand or extend the codebase.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Simulation                              │
│  Orchestrates: Model + Solver + Geometry + TimeController   │
└─────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐
   │ Model   │   │ Solver  │   │ Geometry │   │ Diagnostics│
   │         │   │         │   │          │   │            │
   │compute_ │   │ step()  │   │ grids    │   │ probes     │
   │  rhs()  │   │         │   │ coords   │   │ output     │
   └─────────┘   └─────────┘   └──────────┘   └────────────┘
```

## Data Flow

1. **Initialization**: `Simulation` creates `State` from initial conditions
2. **Time Loop**: For each step:
   - `Model.compute_rhs(state, geometry)` → derivatives
   - `Solver.step(state, rhs, dt)` → new state
   - `TimeController` adjusts dt based on CFL/stability
   - `Diagnostics` record measurements
3. **Output**: Final state + history returned

## Module Structure

```
jax_frc/
├── core/               # Geometry, State, Simulation orchestrator
│   ├── geometry.py     # Grid definitions, coordinate systems
│   ├── state.py        # Simulation state container
│   └── simulation.py   # Main orchestrator
├── models/             # Physics models
│   ├── base.py         # PhysicsModel protocol
│   ├── resistive_mhd.py
│   ├── extended_mhd.py
│   ├── hybrid_kinetic.py
│   ├── neutral_fluid.py
│   └── burning_plasma.py
├── solvers/            # Time integration
│   ├── explicit.py     # RK4, Euler
│   ├── semi_implicit.py
│   └── imex.py         # Implicit-explicit
├── burn/               # Fusion burn physics
│   ├── physics.py      # Reaction rates (Bosch-Hale)
│   ├── species.py      # Fuel tracking
│   └── conversion.py   # Direct energy conversion
├── transport/          # Transport models
│   └── anomalous.py    # Anomalous diffusion
├── comparisons/        # Literature validation
│   └── belova_merging.py
├── validation/         # Validation infrastructure
├── diagnostics/        # Output and probes
├── boundaries/         # Boundary conditions
└── configurations/     # Pre-built setups
```

## Key Abstractions

### PhysicsModel Protocol

All physics models implement this interface:

```python
class PhysicsModel(Protocol):
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all state variables."""
        ...

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return maximum stable timestep."""
        ...
```

### State Container

Immutable dataclass holding all simulation variables:

```python
@dataclass(frozen=True)
class State:
    psi: Array      # Flux function
    v: Array        # Velocity field
    p: Array        # Pressure
    rho: Array      # Density
    t: float        # Current time
```

### Solver Interface

```python
class Solver(Protocol):
    def step(self, state: State, rhs_fn: Callable, dt: float) -> State:
        """Advance state by dt."""
        ...
```

## Extension Points

### Adding a New Model

1. Create `jax_frc/models/your_model.py`
2. Implement `PhysicsModel` protocol
3. Add to `jax_frc/models/__init__.py`
4. Create tests in `tests/test_your_model.py`

See [Adding Models Tutorial](adding-models.md) for walkthrough.

### Adding Diagnostics

1. Create diagnostic class in `jax_frc/diagnostics/`
2. Implement `__call__(state, geometry) -> measurement`
3. Register in `diagnostics/__init__.py`

### Adding Boundary Conditions

1. Add to `jax_frc/boundaries/`
2. Implement `apply(state, geometry) -> state`
3. Make configurable via `from_config()` classmethod
```

**Step 3: Commit**

```bash
git add docs/developer/architecture.md
git commit -m "docs(developer): add architecture overview"
```

---

## Task 2: Create JAX Patterns Guide

**Files:**
- Create: `docs/developer/jax-patterns.md`

**Step 1: Write jax-patterns.md**

Write the following content to `docs/developer/jax-patterns.md`:

```markdown
# JAX Patterns in jax-frc

This guide covers JAX-specific patterns used throughout the codebase. Understanding these is essential for contributing.

## Why JAX?

JAX provides:
- **Automatic differentiation** for gradient-based optimization
- **JIT compilation** for GPU/TPU acceleration
- **Functional transformations** (vmap, pmap) for vectorization

The tradeoff: code must follow functional patterns that enable tracing.

## Core Patterns

### Use `lax.scan` Instead of Python Loops

**Problem**: Python for-loops break JIT tracing.

```python
# BAD - breaks JIT
def time_loop(state, n_steps):
    for _ in range(n_steps):
        state = step(state)
    return state

# GOOD - JIT-compatible
def time_loop(state, n_steps):
    def body(state, _):
        return step(state), None
    final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
    return final_state
```

### Use `lax.cond` Instead of if/else on Traced Values

**Problem**: Python if/else on JAX arrays causes tracing errors.

```python
# BAD - traced value in condition
def apply_bc(psi, use_dirichlet):
    if use_dirichlet:  # Error if use_dirichlet is traced
        return psi.at[0].set(0.0)
    return psi

# GOOD - lax.cond for traced conditions
def apply_bc(psi, use_dirichlet):
    return jax.lax.cond(
        use_dirichlet,
        lambda p: p.at[0].set(0.0),
        lambda p: p,
        psi
    )
```

**When Python if/else IS okay**: Config values known at compile time (use `static_argnums`).

### Static Arguments for JIT

Use `static_argnums` for arguments that:
- Affect array shapes
- Are used in Python control flow
- Are configuration objects

```python
@partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    # geometry.nr, geometry.nz used for shapes - must be static
    ...
```

**Forgetting static_argnums causes**:
- Recompilation on every call (slow)
- Or ConcretizationError (fails)

### Immutable State Updates

JAX arrays are immutable. Use `.at[].set()` for updates:

```python
# BAD - mutation doesn't work
psi[0, :] = 0.0

# GOOD - returns new array
psi = psi.at[0, :].set(0.0)
```

## Debugging JIT Issues

### ConcretizationError

**Symptom**: "Abstract tracer value encountered where concrete value is expected"

**Cause**: Using traced value where Python needs concrete value (if/else, shape, indexing).

**Fix**:
- Use `lax.cond` instead of if/else
- Add argument to `static_argnums`
- Use `jax.debug.print` for debugging

### Shape Errors

**Symptom**: "Shapes must be 1D sequences of concrete values"

**Cause**: Array shape depends on traced value.

**Fix**: Ensure shapes are determined by static arguments only.

### Slow Recompilation

**Symptom**: First call fast, subsequent calls slow.

**Cause**: JIT cache miss due to changing "static" arguments.

**Fix**: Ensure config objects are truly static (same Python object identity).

## Testing JIT Code

Test both eager and compiled paths:

```python
def test_compute_rhs_eager():
    """Test without JIT for easier debugging."""
    with jax.disable_jit():
        result = model.compute_rhs(state, geometry)
    assert result.psi.shape == state.psi.shape

def test_compute_rhs_jit():
    """Test with JIT to catch tracing issues."""
    result = model.compute_rhs(state, geometry)  # JIT enabled
    assert result.psi.shape == state.psi.shape
```

## Common Gotchas

| Issue | Symptom | Solution |
|-------|---------|----------|
| Python loop in JIT | Slow or ConcretizationError | Use `lax.scan` |
| if/else on traced value | ConcretizationError | Use `lax.cond` |
| Missing static_argnums | Recompilation or error | Add to decorator |
| Mutable update | No effect | Use `.at[].set()` |
| Print in JIT | No output | Use `jax.debug.print` |

## Further Reading

- [JAX Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JIT Compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- [Stateful Computations](https://jax.readthedocs.io/en/latest/jax-101/07-state.html)
```

**Step 2: Commit**

```bash
git add docs/developer/jax-patterns.md
git commit -m "docs(developer): add JAX patterns guide"
```

---

## Task 3: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

**Step 1: Write CONTRIBUTING.md**

Write the following content to `CONTRIBUTING.md`:

```markdown
# Contributing to jax-frc

Thank you for your interest in contributing to jax-frc!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/jax-frc.git
cd jax-frc

# Install dependencies
pip install jax jaxlib numpy matplotlib pytest

# Verify installation
py -m pytest tests/ -v -k "not slow"
```

## Code Style

- **Formatting**: Use Black with default settings
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for classes and public methods

```python
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    """Compute time derivatives for state variables.

    Args:
        state: Current simulation state
        geometry: Computational grid

    Returns:
        State with time derivatives in each field
    """
```

## Pull Request Workflow

1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow patterns in existing code
3. **Add tests**: All new functionality needs tests
4. **Run tests**: `py -m pytest tests/ -v`
5. **Commit**: Use conventional commits (`feat:`, `fix:`, `docs:`)
6. **Push**: `git push -u origin feature/your-feature`
7. **Open PR**: Fill in the template

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add D-He3 reaction channel
fix: correct boundary condition at r=0
docs: update Extended MHD equations
refactor: extract resistivity models
test: add conservation law checks
```

## Testing Requirements

### For Bug Fixes

- Add a test that fails without the fix
- Verify the test passes with the fix

### For New Features

- Unit tests for new functions/classes
- Integration test if feature affects simulation flow
- Property-based tests for physics (conservation, bounds)

### For Physics Models

- Conservation law tests (energy, momentum, flux)
- Boundedness tests (positive density, valid temperatures)
- Known solution tests where analytical solutions exist

See [Testing Guide](docs/testing/index.md) for details.

## Architecture

See [Architecture Overview](docs/developer/architecture.md) for:
- Module structure
- Key abstractions (PhysicsModel, Solver, State)
- Extension points

## JAX-Specific Guidelines

See [JAX Patterns](docs/developer/jax-patterns.md) for:
- Using `lax.scan` and `lax.cond`
- Static arguments for JIT
- Debugging tracing issues

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
```

**Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md"
```

---

## Task 4: Update Resistive MHD Doc with Template

**Files:**
- Modify: `docs/models/resistive-mhd.md`

**Step 1: Read current file and identify gaps**

Current file has basic structure but lacks:
- Parameters table
- Detailed implementation mapping
- Validation section

**Step 2: Rewrite with full template**

Replace `docs/models/resistive-mhd.md` with:

```markdown
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
```

**Step 3: Commit**

```bash
git add docs/models/resistive-mhd.md
git commit -m "docs(models): expand resistive-mhd with full template"
```

---

## Task 5: Update Extended MHD Doc with Template

**Files:**
- Modify: `docs/models/extended-mhd.md`

**Step 1: Rewrite with full template**

Replace `docs/models/extended-mhd.md` with:

```markdown
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
```

**Step 2: Commit**

```bash
git add docs/models/extended-mhd.md
git commit -m "docs(models): expand extended-mhd with full template"
```

---

## Task 6: Create Burn Module Doc

**Files:**
- Create: `docs/modules/burn.md`

**Step 1: Create modules directory**

```bash
mkdir -p docs/modules
```

**Step 2: Write burn.md**

Write the following content to `docs/modules/burn.md`:

```markdown
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
```

**Step 3: Commit**

```bash
git add docs/modules/burn.md
git commit -m "docs(modules): add burn module documentation"
```

---

## Task 7: Create Transport Module Doc

**Files:**
- Create: `docs/modules/transport.md`

**Step 1: Write transport.md**

Write the following content to `docs/modules/transport.md`:

```markdown
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
```

**Step 2: Commit**

```bash
git add docs/modules/transport.md
git commit -m "docs(modules): add transport module documentation"
```

---

## Task 8: Create Comparisons Module Doc

**Files:**
- Create: `docs/modules/comparisons.md`

**Step 1: Write comparisons.md**

Write the following content to `docs/modules/comparisons.md`:

```markdown
# Comparisons Module

Validation framework for comparing jax-frc results against published literature.

## Overview

The `jax_frc/comparisons/` module provides infrastructure for:
- Running standardized simulation scenarios
- Comparing results against published data
- Generating validation reports

## Belova et al. FRC Merging Comparison

Primary validation case based on:
> Belova et al., "Numerical study of FRC formation and merging", Physics of Plasmas (2006)

### What's Compared

| Quantity | Description | Agreement Target |
|----------|-------------|------------------|
| Merge time | When two FRCs coalesce | Within 20% |
| Reconnection rate | Peak $d\psi/dt$ | Qualitative trend |
| Energy partition | Thermal vs magnetic | Within 30% |
| Final separatrix | Shape and position | Visual agreement |

### Physics Setup

Two counter-helicity FRCs initialized at opposite ends of domain, allowed to translate and merge:

- **Geometry**: Cylindrical $(r, z)$, typical 0.4m radius × 3m length
- **Initial state**: Hill's vortex equilibrium for each FRC
- **Boundary**: Conducting wall at $r_{max}$, periodic or open in $z$

## Implementation

### Class: `BelovaComparisonSuite`

Location: `jax_frc/comparisons/belova_merging.py`

```python
class BelovaComparisonSuite:
    def run_comparison(
        self,
        geometry: Geometry,
        resistive_config: dict,
        hybrid_config: dict,
    ) -> ComparisonReport:
        """Run both models and compare results."""
```

### Running a Comparison

```python
from jax_frc.comparisons import BelovaComparisonSuite
from jax_frc import Geometry

# Setup
suite = BelovaComparisonSuite()
geometry = suite.create_geometry(nr=64, nz=256)

# Run comparison
report = suite.run_comparison(
    geometry=geometry,
    resistive_config={'resistivity': {'type': 'chodura', 'eta_0': 1e-6}},
    hybrid_config={'n_particles': 100000}
)

# View results
print(report.summary())
```

### Output: `ComparisonReport`

```python
@dataclass
class ComparisonReport:
    resistive_result: MergingResult
    hybrid_result: MergingResult

    def merge_time_difference(self) -> float:
        """Absolute difference in merge times."""

    def merge_time_ratio(self) -> float:
        """Ratio of merge times (resistive/hybrid)."""

    def energy_partition_at_merge(self) -> dict:
        """Energy breakdown at merge time for each model."""

    def summary(self) -> str:
        """Human-readable comparison summary."""
```

### Output: `MergingResult`

```python
@dataclass
class MergingResult:
    times: Array           # Simulation time points
    psi_sep: Array         # Separatrix flux vs time
    E_magnetic: Array      # Magnetic energy vs time
    E_thermal: Array       # Thermal energy vs time
    reconnection_rate: Array  # d(psi)/dt at X-point
    merge_time: float      # Detected merge time
```

## Diagnostics Collected

The suite automatically collects:

1. **Separatrix flux** `psi_sep(t)`: Tracks FRC boundaries
2. **Energy partition** `E_mag(t)`, `E_th(t)`: Conservation check
3. **Reconnection rate** `d(psi)/dt`: Merging dynamics
4. **X-point position** `(r_x, z_x)(t)`: Merging location

## Adding New Comparisons

To add a new literature comparison:

1. Create `jax_frc/comparisons/your_comparison.py`
2. Implement comparison suite class with:
   - `create_geometry()`: Standard geometry for this case
   - `collect_diagnostics()`: What to measure
   - `run_model()`: Run simulation with given config
   - `compare()`: Generate comparison metrics
3. Add tests in `tests/test_your_comparison.py`
4. Document in this file

## Configuration Files

Pre-built configs in `examples/comparisons/`:

```yaml
# belova_resistive.yaml
model:
  type: resistive_mhd
  resistivity:
    type: chodura
    eta_0: 1.0e-6
    eta_anom: 1.0e-3

geometry:
  nr: 64
  nz: 256
  r_max: 0.4
  z_max: 1.5

time:
  t_max: 50.0e-6
  dt_max: 1.0e-8
```

## References

- Belova et al., Physics of Plasmas 13, 056115 (2006)
- Omelchenko & Schaffer, Physics of Plasmas 13, 062111 (2006)
```

**Step 2: Commit**

```bash
git add docs/modules/comparisons.md
git commit -m "docs(modules): add comparisons module documentation"
```

---

## Task 9: Create Adding Models Tutorial

**Files:**
- Create: `docs/developer/adding-models.md`

**Step 1: Write adding-models.md**

Write the following content to `docs/developer/adding-models.md`:

```markdown
# Adding a New Physics Model

This tutorial walks through creating a new physics model for jax-frc.

## Overview

All physics models implement the `PhysicsModel` protocol. We'll create a simple "advection-diffusion" model as an example.

## Step 1: Create the Model File

Create `jax_frc/models/advection_diffusion.py`:

```python
"""Advection-diffusion model for scalar transport."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class AdvectionDiffusion(PhysicsModel):
    """Simple advection-diffusion for a scalar field.

    Solves: ∂φ/∂t + v·∇φ = D∇²φ

    Args:
        diffusivity: Diffusion coefficient D [m²/s]
        velocity: Constant advection velocity (vr, vz) [m/s]
    """

    diffusivity: float
    velocity: tuple[float, float] = (0.0, 0.0)

    @partial(jax.jit, static_argnums=(0, 2))
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute ∂φ/∂t from advection and diffusion."""
        phi = state.psi  # Using psi field for our scalar
        dr, dz = geometry.dr, geometry.dz

        # Diffusion: D∇²φ
        laplacian = self._laplacian(phi, dr, dz)
        diffusion = self.diffusivity * laplacian

        # Advection: -v·∇φ
        dphi_dr = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * dr)
        dphi_dz = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * dz)
        advection = -(self.velocity[0] * dphi_dr + self.velocity[1] * dphi_dz)

        d_phi = diffusion + advection

        return state.replace(psi=d_phi)

    def _laplacian(self, f: jnp.ndarray, dr: float, dz: float) -> jnp.ndarray:
        """Compute ∇²f using central differences."""
        d2f_dr2 = (jnp.roll(f, -1, axis=0) - 2*f + jnp.roll(f, 1, axis=0)) / dr**2
        d2f_dz2 = (jnp.roll(f, -1, axis=1) - 2*f + jnp.roll(f, 1, axis=1)) / dz**2
        return d2f_dr2 + d2f_dz2

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return maximum stable timestep."""
        dr, dz = geometry.dr, geometry.dz
        dx_min = min(dr, dz)

        # Diffusion limit: dt < dx²/(2D)
        dt_diffusion = dx_min**2 / (2 * self.diffusivity) if self.diffusivity > 0 else float('inf')

        # Advection limit: dt < dx/|v|
        v_max = max(abs(self.velocity[0]), abs(self.velocity[1]))
        dt_advection = dx_min / v_max if v_max > 0 else float('inf')

        return min(dt_diffusion, dt_advection)

    @classmethod
    def from_config(cls, config: dict) -> "AdvectionDiffusion":
        """Create model from configuration dictionary."""
        return cls(
            diffusivity=config.get('diffusivity', 1.0),
            velocity=tuple(config.get('velocity', [0.0, 0.0]))
        )
```

## Step 2: Register in __init__.py

Edit `jax_frc/models/__init__.py`:

```python
from jax_frc.models.advection_diffusion import AdvectionDiffusion

__all__ = [
    # ... existing exports ...
    "AdvectionDiffusion",
]
```

## Step 3: Write Tests

Create `tests/test_advection_diffusion.py`:

```python
"""Tests for AdvectionDiffusion model."""

import jax.numpy as jnp
import pytest

from jax_frc import Geometry
from jax_frc.models import AdvectionDiffusion
from jax_frc.core.state import State


@pytest.fixture
def geometry():
    return Geometry(
        coord_system='cartesian',
        nr=32, nz=32,
        r_min=0.0, r_max=1.0,
        z_min=0.0, z_max=1.0
    )


@pytest.fixture
def gaussian_state(geometry):
    """Initial Gaussian profile."""
    r, z = jnp.meshgrid(
        jnp.linspace(0, 1, geometry.nr),
        jnp.linspace(0, 1, geometry.nz),
        indexing='ij'
    )
    psi = jnp.exp(-((r - 0.5)**2 + (z - 0.5)**2) / 0.1)
    return State(psi=psi, v=jnp.zeros((geometry.nr, geometry.nz, 3)),
                 p=jnp.ones((geometry.nr, geometry.nz)),
                 rho=jnp.ones((geometry.nr, geometry.nz)), t=0.0)


class TestAdvectionDiffusion:

    def test_pure_diffusion_smooths(self, geometry, gaussian_state):
        """Pure diffusion should smooth the profile."""
        model = AdvectionDiffusion(diffusivity=1.0, velocity=(0.0, 0.0))

        rhs = model.compute_rhs(gaussian_state, geometry)

        # Center should decrease (diffusing outward)
        center_idx = (geometry.nr // 2, geometry.nz // 2)
        assert rhs.psi[center_idx] < 0

    def test_advection_shifts(self, geometry, gaussian_state):
        """Advection should shift the profile."""
        model = AdvectionDiffusion(diffusivity=0.0, velocity=(1.0, 0.0))

        rhs = model.compute_rhs(gaussian_state, geometry)

        # With positive v_r, should have negative dphi/dt where gradient is positive
        # (profile moving in +r direction)
        assert rhs.psi.shape == gaussian_state.psi.shape

    def test_stable_dt_respects_diffusion(self, geometry, gaussian_state):
        """Stable dt should decrease with higher diffusivity."""
        model_low = AdvectionDiffusion(diffusivity=0.1)
        model_high = AdvectionDiffusion(diffusivity=10.0)

        dt_low = model_low.compute_stable_dt(gaussian_state, geometry)
        dt_high = model_high.compute_stable_dt(gaussian_state, geometry)

        assert dt_high < dt_low

    def test_from_config(self):
        """Factory method should create model correctly."""
        config = {'diffusivity': 2.5, 'velocity': [1.0, -0.5]}
        model = AdvectionDiffusion.from_config(config)

        assert model.diffusivity == 2.5
        assert model.velocity == (1.0, -0.5)

    def test_jit_compilation(self, geometry, gaussian_state):
        """Model should work with JIT."""
        model = AdvectionDiffusion(diffusivity=1.0)

        # First call compiles
        rhs1 = model.compute_rhs(gaussian_state, geometry)
        # Second call uses cached
        rhs2 = model.compute_rhs(gaussian_state, geometry)

        assert jnp.allclose(rhs1.psi, rhs2.psi)
```

## Step 4: Run Tests

```bash
py -m pytest tests/test_advection_diffusion.py -v
```

## Step 5: Document

Add documentation following the [model template](../models/resistive-mhd.md).

## Checklist

Before submitting your new model:

- [ ] Implements `PhysicsModel` protocol (`compute_rhs`, `compute_stable_dt`)
- [ ] Uses `@partial(jax.jit, static_argnums=(0, 2))` on `compute_rhs`
- [ ] Has `from_config()` classmethod for YAML configuration
- [ ] Registered in `jax_frc/models/__init__.py`
- [ ] Has unit tests covering:
  - [ ] Basic physics behavior
  - [ ] Edge cases (zero parameters, etc.)
  - [ ] JIT compilation works
  - [ ] Factory method
- [ ] Has documentation with:
  - [ ] Physics overview
  - [ ] Governing equations
  - [ ] Parameters table
  - [ ] Example usage

## Common Pitfalls

1. **Forgetting `static_argnums`**: Causes recompilation or tracing errors
2. **Python control flow**: Use `lax.cond` not `if/else` on traced values
3. **Mutable operations**: Use `.at[].set()` not direct assignment
4. **Missing `frozen=True`**: Dataclass must be immutable for JIT
```

**Step 2: Commit**

```bash
git add docs/developer/adding-models.md
git commit -m "docs(developer): add tutorial for creating new models"
```

---

## Task 10: Update docs/index.md

**Files:**
- Modify: `docs/index.md`

**Step 1: Update with new sections**

Replace `docs/index.md` with updated content that includes new modules and developer sections:

```markdown
# JAX Plasma Physics Simulation

A JAX-based implementation of plasma physics models for Field-Reversed Configuration (FRC) research.

## Key Features

- **Five Physics Models**: Resistive MHD, Extended MHD, Hybrid Kinetic, Neutral Fluid, Burning Plasma
- **Fusion Burn Physics**: Multi-fuel reactions with direct energy conversion
- **Configuration System**: Pre-built and customizable simulation setups with phase support
- **FRC Merging Simulation**: Two-FRC collision with compression support (Belova et al. validation)
- **Validation Infrastructure**: Automated validation against analytic solutions and references
- **Property-Based Testing**: Physics invariant validation (conservation laws, boundedness)
- **IMEX Solver**: Implicit-explicit time integration for stiff problems

## Documentation

### Getting Started

- [Getting Started](getting-started.md) - Installation and quick start guide

### Physics Models

Detailed documentation for each physics model, including equations, implementation, and parameters:

- [Resistive MHD](models/resistive-mhd.md) - Single-fluid flux function formulation
- [Extended MHD](models/extended-mhd.md) - Two-fluid with Hall effect
- [Hybrid Kinetic](models/hybrid-kinetic.md) - Kinetic ions + fluid electrons
- [Neutral Fluid](models/neutral-fluid.md) - Euler equations for neutrals
- [Burning Plasma](models/burning-plasma.md) - Multi-fuel burn with transport

### Supporting Modules

- [Burn Module](modules/burn.md) - Fusion reaction rates and direct conversion
- [Transport Module](modules/transport.md) - Anomalous particle and energy transport
- [Comparisons Module](modules/comparisons.md) - Literature validation framework

### Developer Guide

For contributors and those extending the codebase:

- [Architecture Overview](developer/architecture.md) - System design and module structure
- [JAX Patterns](developer/jax-patterns.md) - JAX idioms and gotchas
- [Adding Models Tutorial](developer/adding-models.md) - How to create new physics models

### API Reference

- [Core](api/core.md) - Geometry, State, Simulation
- [Solvers](api/solvers.md) - Time integration methods
- [Boundaries](api/boundaries.md) - Boundary conditions
- [Diagnostics](api/diagnostics.md) - Measurement and output
- [Validation](api/validation.md) - Validation infrastructure

### Configurations

- [Configuration System](configurations/index.md) - How configurations work
- [FRC Merging](configurations/merging.md) - Two-FRC collision simulations

### Testing

- [Testing Guide](testing/index.md) - Test suite and how to run tests
- [Invariants](testing/invariants.md) - Property-based physics tests

### Reference

- [Physics Concepts](reference/physics.md) - Key physics background
- [Model Comparison](reference/comparison.md) - Feature comparison table

## Package Structure

```
jax_frc/
├── core/               # Geometry, State, Simulation orchestrator
├── models/             # Physics models (Resistive, Extended, Hybrid, Neutral, Burning)
├── solvers/            # Time integration (Euler, RK4, Semi-implicit, IMEX)
│   └── linear/         # Matrix-free linear solvers (CG)
├── burn/               # Fusion burn physics (reactions, species, conversion)
├── transport/          # Anomalous transport models
├── comparisons/        # Literature validation (Belova et al.)
├── validation/         # Validation infrastructure
├── boundaries/         # Boundary conditions
├── configurations/     # Configuration & phase system
│   └── phases/         # Phase implementations (MergingPhase)
├── diagnostics/        # Probes, output, merging diagnostics
├── equilibrium/        # Equilibrium solvers (Grad-Shafranov, Rigid Rotor)
├── config/             # YAML configuration loading
├── constants.py        # Physical constants
├── operators.py        # Numerical operators
├── results.py          # Result containers
└── input_validation.py # Input validation utilities
```
```

**Step 2: Commit**

```bash
git add docs/index.md
git commit -m "docs: update index with new modules and developer sections"
```

---

## Task 11: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update features and structure**

Update `README.md` to reflect current state. Keep the quick start example, update features and structure:

```markdown
# JAX Plasma Physics Simulation

[![Tests](https://github.com/your-username/jax-frc/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/jax-frc/actions)

A JAX-based GPU-accelerated implementation of plasma physics models for Field-Reversed Configuration (FRC) research.

## Key Features

- **Five Physics Models**: Resistive MHD, Extended MHD, Hybrid Kinetic, Neutral Fluid, Burning Plasma
- **Fusion Burn Physics**: D-T/D-D/D-He3 reactions with Bosch-Hale rates
- **Anomalous Transport**: Configurable particle and energy diffusion
- **Multi-Phase Scenarios**: Automatic phase transitions for complex simulations
- **FRC Merging**: Two-FRC collision with Belova et al. validation
- **Direct Energy Conversion**: Induction-based power recovery modeling
- **Property-Based Testing**: Physics invariant validation

## Quick Start

```bash
pip install jax jaxlib numpy matplotlib
```

```python
from jax_frc import Simulation, Geometry, ResistiveMHD, RK4Solver, TimeController
import jax.numpy as jnp

geometry = Geometry(coord_system='cylindrical', nr=32, nz=64,
                    r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0)
model = ResistiveMHD.from_config({'resistivity': {'type': 'chodura', 'eta_0': 1e-6}})
solver = RK4Solver()
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final_state = sim.run_steps(100)
```

## Documentation

Full documentation is available in the [docs/](docs/index.md) directory:

- [Getting Started](docs/getting-started.md) - Installation and usage
- [Physics Models](docs/models/index.md) - Resistive MHD, Extended MHD, Hybrid Kinetic, Neutral Fluid, Burning Plasma
- [Supporting Modules](docs/modules/burn.md) - Burn physics, transport, comparisons
- [Developer Guide](docs/developer/architecture.md) - Architecture, JAX patterns, extending the code
- [API Reference](docs/api/index.md) - Core, Solvers, Boundaries, Diagnostics
- [Testing](docs/testing/index.md) - Test suite and physics invariants

## Commands

```bash
# Run examples
python run_example.py examples/merging.yaml

# Run tests
py -m pytest tests/ -v

# Run tests (skip slow physics tests)
py -m pytest tests/ -k "not slow"
```

## Project Structure

```
jax-frc/
├── jax_frc/           # Main package
│   ├── models/        # Physics models
│   ├── solvers/       # Time integration
│   ├── burn/          # Fusion burn physics
│   ├── transport/     # Anomalous transport
│   ├── comparisons/   # Literature validation
│   └── ...
├── tests/             # Test suite with invariants
├── examples/          # Example configurations
├── docs/              # Documentation
└── CONTRIBUTING.md    # Contribution guidelines
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR workflow.

## License

This project is provided for educational and research purposes.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with current features and structure"
```

---

## Task 12: Update Testing Docs

**Files:**
- Modify: `docs/testing/index.md`

**Step 1: Update test structure**

Update `docs/testing/index.md` to reflect current tests:

```markdown
# Testing

The project includes a comprehensive test suite with physics invariant validation.

## Running Tests

```bash
# Run all tests
py -m pytest tests/ -v

# Skip slow physics tests
py -m pytest tests/ -k "not slow"

# Run specific model tests
py -m pytest tests/test_resistive_mhd.py -v
py -m pytest tests/test_extended_mhd.py -v
py -m pytest tests/test_hybrid_kinetic.py -v

# Run with coverage
py -m pytest tests/ --cov=jax_frc --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures
├── invariants/                    # Property-based testing
│   ├── boundedness.py             # Value bounds checking
│   ├── conservation.py            # Energy, momentum, flux conservation
│   └── consistency.py             # State consistency checks
├── test_resistive_mhd.py          # Resistive MHD model
├── test_extended_mhd.py           # Extended MHD model
├── test_hybrid_kinetic.py         # Hybrid kinetic model
├── test_neutral_fluid.py          # Neutral fluid model
├── test_burning_plasma.py         # Burning plasma model
├── test_burn_physics.py           # Fusion reaction rates
├── test_transport.py              # Anomalous transport
├── test_boundaries.py             # Boundary conditions
├── test_solvers.py                # Time integration
├── test_scenarios.py              # Multi-phase scenarios
├── test_merging_phase.py          # FRC merging phase
├── test_merging_diagnostics.py    # Merging-specific diagnostics
├── test_merging_integration.py    # End-to-end merging tests
├── test_validation_*.py           # Validation infrastructure
└── test_belova_comparison.py      # Literature comparison
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:
- Operators (laplacian, gradient, divergence)
- Resistivity models
- Boundary condition application

### Model Tests

Test physics models compute correct RHS:
- Shape preservation
- Known limits (zero velocity, uniform fields)
- Stability criteria

### Integration Tests

Test components working together:
- Simulation runs without errors
- Merging scenario completes phases
- Output files written correctly

### Property-Based Tests (Invariants)

Test physics conservation laws:
- Energy conservation (within solver tolerance)
- Flux conservation at boundaries
- Positive density/pressure
- Bounded temperatures

See [Invariants](invariants.md) for detailed documentation.

### Validation Tests

Test against analytical solutions:
- Diffusion equation exact solution
- Alfvén wave propagation
- Equilibrium Grad-Shafranov

### Comparison Tests

Test against published results:
- Belova et al. merging dynamics
- Quantitative metrics within tolerance

## Writing Tests

### For Bug Fixes

```python
def test_bug_123_boundary_at_r_zero():
    """Regression test for issue #123."""
    # Setup that triggered the bug
    geometry = Geometry(r_min=0.0, ...)  # r=0 was problematic

    # This should not raise
    result = model.compute_rhs(state, geometry)

    # Verify correct behavior
    assert jnp.isfinite(result.psi).all()
```

### For New Features

```python
class TestNewFeature:
    def test_basic_functionality(self):
        """Feature works in simple case."""
        ...

    def test_edge_case(self):
        """Feature handles edge case."""
        ...

    def test_integration(self):
        """Feature works with rest of system."""
        ...
```

### For Physics Models

```python
def test_energy_conservation():
    """Total energy should be conserved (within tolerance)."""
    initial_energy = compute_energy(initial_state)
    final_energy = compute_energy(final_state)

    # Allow small numerical drift
    assert abs(final_energy - initial_energy) / initial_energy < 1e-6
```

## Slow Tests

Tests marked `@pytest.mark.slow` run full simulations. Skip with:

```bash
py -m pytest tests/ -k "not slow"
```

These are run in CI but can be skipped locally for faster iteration.
```

**Step 2: Commit**

```bash
git add docs/testing/index.md
git commit -m "docs(testing): update test structure and categories"
```

---

## Task 13: Final Review and Verification

**Step 1: Verify all files exist**

```bash
ls -la docs/developer/
ls -la docs/modules/
cat CONTRIBUTING.md | head -20
```

**Step 2: Build docs if using MkDocs (optional)**

```bash
# If mkdocs is installed
mkdocs build --strict
```

**Step 3: Run tests to ensure nothing broken**

```bash
py -m pytest tests/ -q --tb=no
```

**Step 4: Create summary commit**

```bash
git log --oneline -15
```

---

## Summary

This plan creates/updates 13 files:

| File | Action | Purpose |
|------|--------|---------|
| `docs/developer/architecture.md` | Create | System overview |
| `docs/developer/jax-patterns.md` | Create | JAX idioms guide |
| `docs/developer/adding-models.md` | Create | Model tutorial |
| `CONTRIBUTING.md` | Create | Contribution guide |
| `docs/models/resistive-mhd.md` | Update | Full template |
| `docs/models/extended-mhd.md` | Update | Full template |
| `docs/modules/burn.md` | Create | Burn module docs |
| `docs/modules/transport.md` | Create | Transport module docs |
| `docs/modules/comparisons.md` | Create | Comparisons module docs |
| `docs/index.md` | Update | Add new sections |
| `README.md` | Update | Current features |
| `docs/testing/index.md` | Update | Current test structure |

Each task is one logical change with its own commit.
