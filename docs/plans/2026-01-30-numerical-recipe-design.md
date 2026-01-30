# NumericalRecipe and Scheme Bundles Design

Date: 2026-01-30
Status: Draft (validated in chat)

## Summary

This design separates **physics configuration** from **numerical schemes** by introducing a
runtime `NumericalRecipe` class that owns the stepping process and the scheme bundle
(time integrator, divergence strategy, reconstruction/riemann selection, and stability
control). Physics models remain configurable through `PhysicsModel.create(...)`, while
numerics are consolidated under `NumericalRecipe` and used by `Simulation.step()`.

We also standardize **dimensionless units** (Alfvénic normalization) and make transformation
rules explicit via a dedicated normalization module. Numerical schemes are enumerated as
valid bundles using existing JAX‑FRC APIs (`ResistiveMHD`, `FiniteVolumeMHD`, solver stack).

## Goals

- Make numerical choices explicit and discoverable (no hidden defaults).
- Keep physics configuration separate from numerical scheme selection.
- Enforce dimensionless units and explicit normalization rules.
- Provide a clean path for Resistive‑MHD baseline and later Hall extension.
- Preserve existing model/solver APIs while introducing a single step path.

## Non‑Goals

- Full Hall/Extended MHD scheme integration in this iteration.
- Rewriting existing solvers or models.
- Changing physical equations or adding new physics.

## Key Decisions

1. **Resistive‑MHD baseline** first; Hall extension later.
2. **Alfvénic normalization** is the default; all constants are dimensionless in solver kernels.
3. **`NumericalRecipe` is runtime** and owns the step; `Simulation.step()` delegates to it.
4. **Config controls physics; recipe controls numerics.**

## Architecture Overview

### Current Entry Points (kept)

- `Simulation.from_yaml()` -> `Geometry.from_config()` -> `PhysicsModel.create()` -> `Solver.create()` -> `TimeController`.
- `ResistiveMHD` and `FiniteVolumeMHD` already expose most numerical choices.

### New Runtime Flow

`Simulation.step()` delegates to `NumericalRecipe.step()`:

```
dt = recipe.time_controller.compute_dt(state, model, geometry)
state_next = recipe.solver.step(state, dt, model, geometry)
state_next = recipe.apply_constraints(state_next, model, geometry)
return state_next
```

The recipe becomes the single authority for **stepping cadence** and **constraint policy**.

### Proposed Placement

`jax_frc/solvers/recipe.py` (numerical layer), exported via `jax_frc.solvers`.

## Components and Responsibilities

### NumericalRecipe (runtime, numerics owner)

Fields:
- `solver: Solver` (time integrator)
- `time_controller: TimeController`
- `divergence_strategy: Literal["ct", "clean", "none"]`
- Optional numerical options for FV schemes: `reconstruction`, `limiter_beta`, `riemann_solver`

Responsibilities:
- Compute `dt`
- Advance state using `solver.step(...)`
- Apply divergence strategy (override or delegate to model)
- Validate numerical compatibility of the scheme bundle

### PhysicsModel (physics owner)

No change to `PhysicsModel.create(...)`. Models stay responsible for physics RHS and
`apply_constraints` defaults. NumericalRecipe can override constraint policy if required.

### Normalization Module (dimensionless units)

Add `jax_frc/units/normalization.py` with:
- `NormScales` (base scales: `L0`, `rho0`, `B0`; derived `v0`, `T0`, `p0`, `E0`).
- `to_dimless_state(...)` / `to_physical_state(...)`
- Explicit transformation rules for `eta`, `nu`, `p`, `v`, `B`, `n`, `t`.

Default: **Alfvénic normalization**
- `v0 = B0 / sqrt(rho0)` (mu0=1 in normalized units)
- `T0 = L0 / v0`, `p0 = rho0 * v0^2`

All solver kernels operate in dimensionless units; only IO/diagnostics use physical units.

## Reference‑Driven Numerical Patterns (Mapped to APIs)

The design aligns with AGATE/gPLUTO/Athena practices by keeping the **numerical pipeline
explicit** in configuration and selecting schemes by name:

- Reconstruction: `solvers.riemann.reconstruction.reconstruct_plm` (MC beta = 1.3 default).
- Riemann solvers: `FiniteVolumeMHD(riemann_solver="hll"|"hlld"|"ct_hlld")`.
- Divergence control: CT (`constrained_transport.induction_rhs_ct`) or
  projection cleaning (`divergence_cleaning.clean_divergence`).
- Stability: centralized via `TimeController` + `model.compute_stable_dt()`.

## Numerical Scheme Bundles (Valid Combinations)

| Bundle | Model config | Solver config | Key numeric pieces | Divergence handling | Notes |
|---|---|---|---|---|---|
| Resistive‑CT (baseline) | `type: resistive_mhd`, `advection_scheme: ct`, `normalized_units: true` | `type: rk4` | CT induction | Exact (CT) | Default |
| Resistive‑SkewSym | `advection_scheme: skew_symmetric` | `type: rk4` | Skew‑symmetric induction | Exact | Energy‑friendly |
| Resistive‑Central + Clean | `advection_scheme: central` | `type: rk4` | Central diff | Projection cleaning | Simple |
| Resistive‑HLL (AGATE‑style) | `advection_scheme: hll`, `hll_beta: 1.3` | `type: rk4` | PLM + HLL flux | Projection cleaning | AGATE‑compatible |
| Godunov‑HLL | `type: finite_volume_mhd`, `riemann_solver: hll`, `reconstruction: plm` | `type: euler` | PLM + HLL update | Optional | Robust |
| Godunov‑HLLD | `riemann_solver: hlld`, `reconstruction: plm` | `type: euler` | PLM + HLLD update | Optional | Higher fidelity |
| Godunov‑CT‑HLLD | `riemann_solver: ct_hlld`, `reconstruction: plm` | `type: euler` | CT‑HLLD update | CT‑style | Divergence aware |
| Resistive‑IMEX | `type: resistive_mhd`, `advection_scheme: ct` | `type: imex` | Explicit hyperbolic + implicit diffusion | CT or clean | For stiff eta |

## Configuration Patterns

Example: **Resistive‑CT baseline**
```
model:
  type: resistive_mhd
  advection_scheme: ct
  normalized_units: true
solver:
  type: rk4
time:
  cfl_safety: 0.4
```

Example: **Finite‑Volume HLLD**
```
model:
  type: finite_volume_mhd
  riemann_solver: hlld
  reconstruction: plm
  limiter_beta: 1.3
solver:
  type: euler
```

## Data Flow (Stepping)

```
Simulation.step()
  -> NumericalRecipe.step()
       -> dt = time_controller.compute_dt(...)
       -> state_next = solver.step(state, dt, model, geometry)
       -> state_next = recipe.apply_constraints(state_next, model, geometry)
       -> return state_next
```

## Error Handling and Validation

- `Solver.step_checked` guards NaN/Inf; recipe can opt into checked stepping for debug runs.
- `NumericalRecipe.validate()` should assert bundle compatibility (e.g., CT only with CT‑capable paths).
- Divergence cleaning should be optional and avoid double‑cleaning when CT is used.

## Testing Strategy

Focus: validate that **numerics remain stable, conservative, and reproducible** across
scheme bundles and normalization modes.

### Required coverage for NumericalRecipe changes
- **Recipe step unit tests**: `NumericalRecipe.step()` advances time, uses `TimeController`,
  and applies the chosen constraint strategy exactly once.
- **Determinism tests**: same inputs + same config produce bitwise‑stable outputs
  (for CPU runs or with a documented tolerance for GPU).

### Normalization and unit transforms
- **Round‑trip state transforms**: `state -> dimless -> physical` matches original
  within tolerance for `n`, `p`, `v`, `B`, `t`.
- **Coefficient scaling**: verify `eta* = eta / (L0 * v0)` and `nu* = nu / (L0 * v0)`
  against known reference values.

### Scheme‑bundle smoke matrix
Every bundle in the table must execute **at least one step** on a small grid:
- `ResistiveMHD` + `ct`, `skew_symmetric`, `central`, `hll`
- `FiniteVolumeMHD` + `hll`, `hlld`, `ct_hlld`
- `imex` path for stiff resistive diffusion

### Invariants and physics sanity checks
- **Divergence control**: `||∇·B||_2` remains bounded; CT keeps it at machine‑level.
- **Positivity**: density and pressure floors enforced (no negative values).
- **Energy behavior**: track total energy drift (bounded, no explosive growth).

### Regression and convergence
- **Canonical problems**: Orszag‑Tang vortex and Alfvén wave compare to stored curves.
- **Grid refinement**: at least one case demonstrating expected error decrease when
  doubling resolution.

### Failure diagnostics
- On failure, record: `dt`, max wave speed, max `|∇·B|`, min density/pressure,
  and NaN/Inf detection details to accelerate debugging.

### Validation gates
- **Routine changes**: `py -m pytest tests/ -k "not slow"`.
- **Major numerics changes**: `py -m scripts/run_validation.py --all` plus at least
  one documented physics validation case.

## Open Questions

- Should `NumericalRecipe` own `apply_constraints` unconditionally or defer to `model.apply_constraints`?
- Should recipes be serialized to configs for reproducibility metadata?
- Should `FiniteVolumeMHD` allow `rk4` for higher‑order time integration?
