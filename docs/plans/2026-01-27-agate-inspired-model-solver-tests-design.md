# AGATE-Inspired Improvements for JAX-FRC (Model, Solver, Tests)

Date: 2026-01-27

## Goals
- Adopt AGATE-style component separation without a full refactor.
- Improve model fidelity, solver stability/performance, and test/validation rigor.
- Preserve JAX-first patterns and existing API surface where possible.

## Model Design (Incremental Componentization)
- Extend `PhysicsModel` to support `default()`, `query_config()`, and `validate_state()` hooks.
- Split `compute_rhs()` into internal component calls: closures, sources, constraints.
- Promote `apply_constraints()` to a first-class physics step: divB handling, temperature floors, BCs.

Target files:
- `jax_frc/models/base.py`
- `jax_frc/models/resistive_mhd.py`
- `jax_frc/models/extended_mhd.py`

## Solver Design (Evolver-Like Orchestration)
- Add a light precheck/step/postcheck wrapper (AGATE Evolver pattern) in `Simulation.step()`
  or `Solver.step()` to integrate model validation and constraints.
- Introduce time-stepper classes with `default()` and `query_config()` for explicit, SSP, and IMEX.
- Integrate divergence cleaning as an optional solver hook (frequency + tolerance).
- Add a `lax.scan` path for multi-step runs to reduce Python overhead.
- Extend `TimeController` to include solver-specific stability limits (IMEX cfl factor, tol).

Target files:
- `jax_frc/core/simulation.py`
- `jax_frc/solvers/base.py`
- `jax_frc/solvers/explicit.py`
- `jax_frc/solvers/imex.py`
- `jax_frc/solvers/time_controller.py`
- `jax_frc/solvers/divergence_cleaning.py`

## Test and Validation Design
- Expand analytic validation to compare solver order/consistency (Euler, RK4, IMEX).
- Add component tests for Hall/electron pressure term toggles and limiting behavior.
- Apply energy/flux invariants in model tests using existing invariant utilities.
- Add divergence-cleaning regression tests (divB drop and energy stability).

Target files:
- `tests/test_imex_diffusion.py`
- `tests/test_extended_mhd_3d.py`
- `tests/test_resistive_mhd_3d.py`
- `tests/invariants/conservation.py`
- `tests/test_divergence_cleaning_3d.py`

## Success Criteria
- All new tests pass with current CI configuration.
- No change to public APIs required for baseline usage.
- Clear configurability and reproducibility via `query_config()` and defaults.
