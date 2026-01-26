# JAX Plasma Physics Simulation

A JAX-based implementation of three primary plasma physics models used in Field-Reversed Configuration (FRC) research: Resistive MHD (Lamy Ridge), Extended MHD (NIMROD), and Hybrid Kinetic-Fluid (HYM).

## Key Features

- **Three Physics Models**: Resistive MHD, Extended MHD, and Hybrid Kinetic with increasing fidelity
- **Configuration System**: Pre-built and customizable simulation setups with phase support
- **FRC Merging Simulation**: Two-FRC collision with compression support (Belova et al. validation)
- **Validation Infrastructure**: Automated validation against analytic solutions and references
- **Property-Based Testing**: Physics invariant validation (conservation laws, boundedness)
- **IMEX Solver**: Implicit-explicit time integration for stiff problems

## Documentation

- [Getting Started](getting-started.md) - Installation and quick start guide
- [Physics Models](models/index.md) - Overview of available physics models
  - [Resistive MHD](models/resistive-mhd.md)
  - [Extended MHD](models/extended-mhd.md)
  - [Hybrid Kinetic](models/hybrid-kinetic.md)
- [API Reference](api/index.md)
  - [Core](api/core.md) - Geometry, State, Simulation
  - [Solvers](api/solvers.md) - Time integration methods
  - [Boundaries](api/boundaries.md) - Boundary conditions
  - [Diagnostics](api/diagnostics.md) - Measurement and output
  - [Validation](api/validation.md) - Validation infrastructure
- [Configurations](configurations/index.md) - Simulation configuration system
  - [FRC Merging](configurations/merging.md) - Two-FRC collision simulations
- [Testing](testing/index.md) - Test suite and validation
  - [Invariants](testing/invariants.md) - Property-based physics tests
- [Reference](reference/index.md)
  - [Physics Concepts](reference/physics.md) - Key physics background
  - [Model Comparison](reference/comparison.md) - Feature comparison table

## Package Structure

```
jax_frc/
├── core/               # Geometry, State, Simulation orchestrator
├── models/             # Physics models (Resistive, Extended, Hybrid)
├── solvers/            # Time integration (Euler, RK4, Semi-implicit, IMEX)
│   └── linear/         # Matrix-free linear solvers (CG)
├── boundaries/         # Boundary conditions
├── configurations/     # Configuration & phase system
│   └── phases/         # Phase implementations (MergingPhase)
├── validation/         # Validation infrastructure
├── diagnostics/        # Probes, output, merging diagnostics
├── equilibrium/        # Equilibrium solvers (Grad-Shafranov, Rigid Rotor)
├── config/             # YAML configuration loading
├── constants.py        # Physical constants
├── operators.py        # Numerical operators
├── results.py          # Result containers
└── input_validation.py # Input validation utilities
```
