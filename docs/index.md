# JAX Plasma Physics Simulation

A JAX-based implementation of three primary plasma physics models used in Field-Reversed Configuration (FRC) research: Resistive MHD (Lamy Ridge), Extended MHD (NIMROD), and Hybrid Kinetic-Fluid (HYM).

## Key Features

- **Three Physics Models**: Resistive MHD, Extended MHD, and Hybrid Kinetic with increasing fidelity
- **Multi-Phase Scenarios**: Run complex simulations with automatic phase transitions
- **FRC Merging Simulation**: Two-FRC collision with compression support (Belova et al. validation)
- **Modular OOP Framework**: Swappable physics models, solvers, and diagnostics
- **Property-Based Testing**: Physics invariant validation (conservation laws, boundedness)
- **Input Validation**: Runtime validation with helpful error messages

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
- [Scenarios](scenarios/index.md) - Multi-phase simulation framework
  - [FRC Merging](scenarios/merging.md) - Two-FRC collision simulations
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
├── solvers/            # Time integration (Euler, RK4, Semi-implicit)
├── boundaries/         # Boundary conditions
├── equilibrium/        # Equilibrium solvers (Grad-Shafranov, Rigid Rotor)
├── scenarios/          # Multi-phase scenario framework
├── diagnostics/        # Probes, output, and merging diagnostics
├── config/             # YAML configuration loading
├── constants.py        # Physical constants
├── operators.py        # Numerical operators
├── results.py          # Result containers
└── validation.py       # Input validation
```
