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
