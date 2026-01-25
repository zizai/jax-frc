# JAX-FRC Simulation Framework Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade, object-oriented FRC plasma simulation framework with accurate physics and flexible configuration.

**Architecture:** Modular component-based design with swappable physics plugins, YAML configuration, and multi-model coupling.

**Tech Stack:** JAX, Python 3.10+, HDF5, YAML

---

## 1. Core Architecture

The framework centers on a `Simulation` class that orchestrates independent, swappable components. Each component has a well-defined interface and can be configured or replaced without affecting others.

### Core Classes

```
Simulation          - Main orchestrator, owns all components, runs time loop
├── Geometry        - Defines mesh, coordinate system, domain bounds
├── State           - Container for all physical quantities (B, E, n, p, v, particles)
├── Solver          - Time integration strategy (explicit, semi-implicit, subcycling)
└── PhysicsModel    - Abstract base for MHD, Extended MHD, Hybrid models
```

### Plugin Components (swappable)

```
BoundaryCondition   - Wall, periodic, open, flux-conserving
TransportModel      - Classical, anomalous, neoclassical
SourceModel         - Neutral beams, RF heating, fueling
EquilibriumSolver   - Grad-Shafranov, force-free, analytic profiles
Diagnostics         - Output quantities, synthetic measurements
```

### Design Principles

1. **Immutable State** - JAX works best with immutable data. `State` is a frozen dataclass that gets replaced each timestep, not mutated.

2. **Configuration over code** - Simulations defined via config files (YAML/JSON), not by editing Python.

3. **Lazy initialization** - Components created from config but not allocated until `simulation.initialize()` called.

4. **Device-agnostic** - Components don't know if they're on CPU, GPU, or distributed. JAX handles placement.

---

## 2. Geometry and State

### Geometry Class

Handles the computational domain and coordinate transformations. FRC simulations typically use cylindrical coordinates with axisymmetry.

```python
@dataclass(frozen=True)
class Geometry:
    coord_system: str          # "cylindrical", "cartesian"
    nr: int                    # Radial grid points
    nz: int                    # Axial grid points
    r_min: float              # Inner radius (>0 to avoid singularity)
    r_max: float              # Outer radius (wall location)
    z_min: float              # Axial extent
    z_max: float

    # Computed properties
    r: Array                   # 1D radial coordinates
    z: Array                   # 1D axial coordinates
    dr: float                  # Grid spacing
    dz: float
    cell_volumes: Array        # 2D array of cell volumes (includes 2πr factor)
```

### State Class

Single container for all physical quantities. Using JAX's `pytree` registration allows entire state to flow through JIT-compiled functions.

```python
@dataclass(frozen=True)
class State:
    # Fields (all shape: nr × nz or nr × nz × 3 for vectors)
    psi: Array                 # Poloidal flux function
    B: Array                   # Magnetic field vector
    E: Array                   # Electric field vector
    n: Array                   # Number density
    p: Array                   # Pressure (scalar or tensor)
    v: Array                   # Fluid velocity

    # Particles (for hybrid model, None otherwise)
    particles: Optional[ParticleState]

    # Metadata
    time: float
    step: int
```

### ParticleState

For hybrid kinetic simulations:

```python
@dataclass(frozen=True)
class ParticleState:
    x: Array          # Positions (n_particles, 3)
    v: Array          # Velocities (n_particles, 3)
    w: Array          # Delta-f weights (n_particles,)
    species: str      # "ion", "beam", etc.
```

---

## 3. Physics Models

The three simulation types share a common interface but implement different physics.

### Abstract Base

```python
class PhysicsModel(ABC):
    @abstractmethod
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all evolved quantities."""
        pass

    @abstractmethod
    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return CFL-stable timestep for this model."""
        pass

    @abstractmethod
    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Enforce physical constraints (e.g., div(B)=0)."""
        pass
```

### Resistive MHD

Single-fluid MHD with resistive Ohm's law: **E + v×B = ηJ**

```python
class ResistiveMHD(PhysicsModel):
    resistivity: ResistivityModel  # Spitzer, Chodura, or custom

    def compute_rhs(self, state, geometry):
        # Grad-Shafranov evolution: ∂ψ/∂t + v·∇ψ = (η/μ₀)Δ*ψ
        # Circuit coupling for external coils
        pass

    def compute_stable_dt(self, state, geometry):
        # Diffusion CFL: dt < dx² / (4D) where D = η/μ₀
        pass
```

### Extended MHD

Two-fluid with Hall term: **E = -v×B + ηJ + (J×B)/(ne) - ∇pₑ/(ne)**

```python
class ExtendedMHD(PhysicsModel):
    hall_enabled: bool
    electron_pressure: bool
    halo_model: HaloModel          # Vacuum region handling

    def compute_rhs(self, state, geometry):
        # Extended Ohm's law with Hall and electron pressure terms
        pass

    def compute_stable_dt(self, state, geometry):
        # Whistler CFL: dt < dx / v_whistler where v_w = k*B/(μ₀*n*e)
        pass
```

### Hybrid Kinetic

Kinetic ions + fluid electrons:

```python
class HybridKinetic(PhysicsModel):
    particle_pusher: ParticlePusher  # Boris, Vay, etc.
    delta_f: bool                     # Full-f or delta-f method
    equilibrium: EquilibriumDistribution  # f₀ for delta-f

    def compute_rhs(self, state, geometry):
        # Boris push: dv/dt = (q/m)(E + v×B)
        # Weight evolution: dw/dt = -(1-w) d ln f₀/dt
        # Field solve from fluid electrons
        pass

    def compute_stable_dt(self, state, geometry):
        # Cyclotron CFL: dt < 0.1 / ω_ci
        pass
```

### Multi-Model Coupling

```python
class ModelCoupler:
    def mhd_to_hybrid(self, mhd_state: State, n_particles: int) -> State:
        """Sample particles from MHD fields to initialize hybrid run."""

    def hybrid_to_mhd(self, hybrid_state: State) -> State:
        """Compute fluid moments from particle distribution."""
```

---

## 4. Plugin Components

### Boundary Conditions

```python
class BoundaryCondition(ABC):
    @abstractmethod
    def apply(self, state: State, geometry: Geometry) -> State:
        """Modify state at domain boundaries."""
        pass

class ConductingWall(BoundaryCondition):
    """Perfect conductor: B_tangent = 0, E_normal = 0"""

class FluxConserving(BoundaryCondition):
    """Maintains total magnetic flux through boundaries"""

class OpenBoundary(BoundaryCondition):
    """Outflow conditions for translation simulations"""

class SymmetryAxis(BoundaryCondition):
    """Handles r=0 axis singularity in cylindrical coords"""
```

### Transport Models

```python
class TransportModel(ABC):
    def compute_diffusion(self, state: State, geometry: Geometry) -> Dict[str, Array]:
        """Returns diffusive fluxes for each transported quantity."""

class ClassicalTransport(TransportModel):
    """Spitzer-Braginskii collisional transport"""

class AnomalousTransport(TransportModel):
    """Bohm-like or gyro-Bohm scaling for turbulent transport"""
    chi_multiplier: float  # Tunable coefficient
```

### Source Models

```python
class SourceModel(ABC):
    def compute_sources(self, state: State, geometry: Geometry, time: float) -> State:
        """Returns source terms to add to RHS."""

class NeutralBeamInjection(SourceModel):
    """NBI heating and fueling for FRC"""
    power: float           # Beam power [W]
    energy: float          # Beam energy [eV]
    injection_angle: float

class OhmicHeating(SourceModel):
    """Resistive heating from current dissipation"""
```

---

## 5. Solver and Time Integration

### Solver Interface

```python
class Solver(ABC):
    @abstractmethod
    def step(self, state: State, dt: float, model: PhysicsModel) -> State:
        """Advance state by one timestep."""
        pass

class ExplicitRK4(Solver):
    """Standard 4th-order Runge-Kutta for non-stiff problems."""

class SemiImplicit(Solver):
    """Implicit treatment of stiff terms (Whistler waves, diffusion)."""
    implicit_terms: List[str]  # Which terms to treat implicitly

class SubcyclingSolver(Solver):
    """Subcycles fast physics within slower outer timestep."""
    inner_solver: Solver
    outer_solver: Solver
    subcycle_criterion: Callable  # When to subcycle
```

### Adaptive Timestepping

```python
@dataclass
class TimeController:
    cfl_safety: float = 0.5        # Safety factor for CFL
    dt_min: float = 1e-12          # Minimum allowed timestep
    dt_max: float = 1e-3           # Maximum allowed timestep

    def compute_dt(self, state: State, model: PhysicsModel, geometry: Geometry) -> float:
        """Compute stable timestep from all constraints."""
        dt_cfl = model.compute_stable_dt(state, geometry)
        dt_transport = self._transport_dt(state, geometry)
        dt_source = self._source_dt(state)

        dt = self.cfl_safety * min(dt_cfl, dt_transport, dt_source)
        return jnp.clip(dt, self.dt_min, self.dt_max)
```

---

## 6. Equilibrium and Initialization

### Equilibrium Solvers

```python
class EquilibriumSolver(ABC):
    @abstractmethod
    def solve(self, geometry: Geometry, constraints: EquilibriumConstraints) -> State:
        """Compute self-consistent equilibrium state."""
        pass

class GradShafranovSolver(EquilibriumSolver):
    """Iterative solution of Grad-Shafranov equation.

    Δ*ψ = -μ₀ r² dp/dψ - F dF/dψ
    """
    max_iterations: int = 1000
    tolerance: float = 1e-8

class RigidRotorEquilibrium(EquilibriumSolver):
    """Analytic FRC equilibrium with rigid rotation."""
    rotation_frequency: float  # Ω for v_θ = Ω*r

class ImportedEquilibrium(EquilibriumSolver):
    """Load equilibrium from external file (EQDSK, HDF5)."""
    file_path: str
    format: str
```

### Equilibrium Constraints

```python
@dataclass
class EquilibriumConstraints:
    total_current: float           # Plasma current [A]
    separatrix_radius: float       # FRC radius [m]
    elongation: float              # Length / diameter
    pressure_profile: str          # "peaked", "flat", "hollow"
    peak_beta: float               # β = p / (B²/2μ₀)
    external_field: float          # Applied axial field [T]
```

---

## 7. Simulation Orchestration

### Simulation Class

```python
class Simulation:
    def __init__(self, config: SimulationConfig):
        self.geometry = Geometry.from_config(config.geometry)
        self.model = PhysicsModel.create(config.model)
        self.solver = Solver.create(config.solver)
        self.boundaries = [BC.create(bc) for bc in config.boundaries]
        self.transport = TransportModel.create(config.transport)
        self.sources = [Source.create(s) for s in config.sources]
        self.diagnostics = Diagnostics(config.diagnostics)
        self.time_controller = TimeController(**config.time)

        self.state: Optional[State] = None

    def initialize(self, equilibrium: EquilibriumSolver = None,
                   initial_state: State = None):
        """Set up initial state from equilibrium or provided state."""
        if equilibrium:
            self.state = equilibrium.solve(self.geometry, self.constraints)
        else:
            self.state = initial_state

    def run(self, t_end: float, checkpoint_interval: float = None) -> State:
        """Run simulation until t_end, optionally saving checkpoints."""
        while self.state.time < t_end:
            dt = self.time_controller.compute_dt(self.state, self.model, self.geometry)
            self.state = self._advance(dt)
            self.diagnostics.record(self.state)
        return self.state

    def _advance(self, dt: float) -> State:
        """Single timestep: sources → physics → boundaries → constraints."""
        state = self.state
        for source in self.sources:
            state = source.compute_sources(state, self.geometry, state.time)
        state = self.solver.step(state, dt, self.model)
        for bc in self.boundaries:
            state = bc.apply(state, self.geometry)
        state = self.model.apply_constraints(state, self.geometry)
        return state
```

---

## 8. Configuration System

### YAML Configuration Format

```yaml
# frc_baseline.yaml
geometry:
  coord_system: cylindrical
  nr: 128
  nz: 256
  r_min: 0.01
  r_max: 0.4        # 40 cm radius
  z_min: -1.0
  z_max: 1.0        # 2 m length

model:
  type: extended_mhd
  hall_enabled: true
  electron_pressure: true
  resistivity:
    type: chodura
    eta_0: 1e-6
    eta_anom: 1e-3
    threshold: 1e4

equilibrium:
  type: grad_shafranov
  total_current: 500e3    # 500 kA
  separatrix_radius: 0.25
  peak_beta: 0.9
  external_field: 0.1     # 1000 Gauss

boundaries:
  - type: symmetry_axis
    location: r_min
  - type: conducting_wall
    location: r_max
  - type: open_boundary
    location: [z_min, z_max]

sources:
  - type: neutral_beam
    power: 2e6            # 2 MW
    energy: 15e3          # 15 keV

time:
  dt_max: 1e-6
  cfl_safety: 0.4

output:
  checkpoint_interval: 1e-4
  diagnostics: [magnetic_flux, beta, energy, separatrix]
```

### Usage

```python
sim = Simulation.from_yaml("frc_baseline.yaml")
sim.initialize()
final_state = sim.run(t_end=1e-3)
```

---

## 9. Diagnostics and Output

### Diagnostics System

```python
class Diagnostics:
    def __init__(self, config: DiagnosticsConfig):
        self.probes = [Probe.create(p) for p in config.probes]
        self.history = SimulationHistory()
        self.output_dir = config.output_dir

    def record(self, state: State):
        """Called each timestep to collect diagnostic data."""
        for probe in self.probes:
            probe.measure(state, self.history)

    def save_checkpoint(self, state: State, filename: str):
        """Save full state for restart or analysis."""

    def export_summary(self) -> Dict:
        """Return key metrics for parameter optimization."""
```

### Built-in Diagnostics

```python
class MagneticFluxProbe(Probe):
    """Track poloidal flux at specified locations."""

class SeparatrixTracker(Probe):
    """Find and track FRC separatrix (ψ = 0 contour)."""
    # Reports: x-point location, radius, elongation, volume

class EnergyBalance(Probe):
    """Track energy components and conservation."""
    # Magnetic, thermal, kinetic, source input, losses

class BetaProfile(Probe):
    """Compute local and volume-averaged beta."""

class ConfinementTime(Probe):
    """Estimate particle and energy confinement times."""

class SyntheticMagnetics(Probe):
    """Simulate B-dot probe signals at specified locations."""
    probe_positions: List[Tuple[float, float]]
```

### Output Formats

- **HDF5** for full state checkpoints (restart-capable)
- **CSV/JSON** for time histories (easy plotting)
- **VTK** for 3D visualization in ParaView

---

## 10. Testing and Validation

### Testing Hierarchy

```python
# Unit tests - individual components
class TestGradShafranovSolver:
    def test_solovev_analytic(self):
        """Compare against known analytic Solov'ev solution."""

    def test_flux_conservation(self):
        """Verify total flux preserved during iteration."""

# Integration tests - component interactions
class TestResistiveMHDSimulation:
    def test_diffusion_rate(self):
        """Verify resistive decay matches analytic τ_R."""

    def test_reconnection_scaling(self):
        """Check Sweet-Parker scaling in reconnection region."""

# Physics validation - against published results
class TestFRCBenchmarks:
    def test_tilt_stability_threshold(self):
        """Compare tilt mode growth rate vs. s parameter."""

    def test_rotational_stabilization(self):
        """Verify FLR stabilization at high rotation."""
```

### Invariant Testing

Integrate with existing infrastructure:

```python
class ProductionInvariants:
    boundedness = [FiniteValues(), PositiveValues("n"), BoundedRange("beta", 0, 2)]
    conservation = [EnergyConservation(rtol=0.01), FluxConservation(rtol=0.001)]
    consistency = [DivergenceFreeB(atol=1e-10)]
```

### Validation Database

```yaml
# validation/frc_experiments.yaml
- name: "C2-W Shot 12345"
  reference: "Binderbauer et al., PoP 2015"
  metrics:
    confinement_time: 5e-3
    peak_beta: 0.9
    separatrix_radius: 0.35
```

---

## File Structure

```
jax_frc/
├── __init__.py
├── core/
│   ├── geometry.py          # Geometry class
│   ├── state.py             # State, ParticleState
│   └── simulation.py        # Simulation orchestrator
├── models/
│   ├── base.py              # PhysicsModel ABC
│   ├── resistive_mhd.py
│   ├── extended_mhd.py
│   ├── hybrid_kinetic.py
│   └── coupler.py           # ModelCoupler
├── solvers/
│   ├── base.py              # Solver ABC
│   ├── explicit.py          # RK4, etc.
│   ├── semi_implicit.py
│   └── subcycling.py
├── boundaries/
│   ├── base.py
│   ├── conducting.py
│   ├── open.py
│   └── symmetry.py
├── transport/
│   ├── base.py
│   ├── classical.py
│   └── anomalous.py
├── sources/
│   ├── base.py
│   ├── neutral_beam.py
│   └── ohmic.py
├── equilibrium/
│   ├── base.py
│   ├── grad_shafranov.py
│   ├── rigid_rotor.py
│   └── imported.py
├── diagnostics/
│   ├── base.py
│   ├── probes.py
│   └── output.py
├── config/
│   ├── loader.py            # YAML/JSON parsing
│   └── schema.py            # Validation
└── tests/
    ├── unit/
    ├── integration/
    └── validation/
```

---

## Implementation Priority

1. **Core infrastructure** - Geometry, State, Simulation shell
2. **Resistive MHD** - Simplest model, validate architecture
3. **Configuration system** - Enable YAML-driven runs
4. **Extended MHD** - Add Hall physics
5. **Equilibrium solvers** - Grad-Shafranov
6. **Hybrid Kinetic** - Most complex, needs stable foundation
7. **Diagnostics** - Production output formats
8. **Multi-model coupling** - MHD → Hybrid handoff
