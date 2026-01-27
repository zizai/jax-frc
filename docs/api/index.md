# API Reference

The `jax_frc` package provides a modular OOP framework for building and running FRC simulations with swappable physics models, solvers, and diagnostics.

## Core Modules

- [Core](core.md) - Geometry, State, and Simulation classes
- [Solvers](solvers.md) - Time integration methods
- [Boundaries](boundaries.md) - Boundary conditions
- [Diagnostics](diagnostics.md) - Measurement and output
- [Validation](validation.md) - Validation infrastructure

## Package Structure

```
jax_frc/
├── core/
│   ├── geometry.py        # Computational domain and coordinates
│   ├── state.py           # State containers (State, ParticleState)
│   └── simulation.py      # Main orchestrator class
├── models/
│   ├── base.py            # PhysicsModel abstract base class
│   ├── resistive_mhd.py   # Single-fluid resistive MHD
│   ├── extended_mhd.py    # Two-fluid Extended MHD with Hall term
│   ├── hybrid_kinetic.py  # Hybrid kinetic with delta-f PIC
│   ├── particle_pusher.py # Boris particle pusher
│   ├── resistivity.py     # Spitzer and Chodura resistivity models
│   └── energy.py          # Thermal transport
├── solvers/
│   ├── base.py            # Solver abstract base class
│   ├── explicit.py        # Euler and RK4 solvers
│   ├── semi_implicit.py   # Semi-implicit and hybrid solvers
│   ├── imex.py            # IMEX (Implicit-Explicit) solver
│   ├── time_controller.py # Adaptive timestep control
│   ├── divergence_cleaning.py # ∇·B cleaning
│   └── linear/
│       └── conjugate_gradient.py # Matrix-free CG solver
├── boundaries/
│   ├── base.py            # BoundaryCondition base class
│   ├── conducting.py      # Conducting wall boundary
│   ├── symmetry.py        # Symmetry axis boundary
│   └── time_dependent.py  # Time-dependent mirror BC for compression
├── configurations/
│   ├── base.py            # AbstractConfiguration base class
│   ├── linear_configuration.py # YAML-driven configuration
│   ├── frc_merging.py     # Belova merging configurations
│   ├── analytic.py        # Analytic test cases
│   ├── phase.py           # Phase base class and registry
│   ├── transitions.py     # Transition conditions
│   └── phases/
│       └── merging.py     # FRC merging phase
├── validation/
│   ├── metrics.py         # Error metrics (L2, Linf, RMSE)
│   ├── references.py      # Reference data management
│   ├── result.py          # ValidationResult container
├── equilibrium/
│   ├── base.py            # Equilibrium base class
│   ├── grad_shafranov.py  # Grad-Shafranov solver
│   └── rigid_rotor.py     # Rigid rotor equilibrium
├── diagnostics/
│   ├── probes.py          # Diagnostic probes (Flux, Energy, Beta, Current)
│   ├── merging.py         # Merging-specific diagnostics
│   └── output.py          # HDF5 checkpoints and time history I/O
├── config/
│   └── loader.py          # YAML configuration loading
├── constants.py           # Physical constants (MU0, QE, ME, MI, KB)
├── operators.py           # Numerical operators (grad, div, curl, laplacian)
├── results.py             # Result containers for simulations
└── input_validation.py    # Input validation utilities
```

## Physical Constants

```python
MU0 = 1.2566e-6   # Permeability of free space
QE = 1.602e-19    # Elementary charge
ME = 9.109e-31    # Electron mass
MI = 1.673e-27    # Ion mass (proton)
KB = 1.381e-23    # Boltzmann constant
EPSILON0 = 8.854e-12  # Permittivity of free space
C = 2.998e8       # Speed of light
```
