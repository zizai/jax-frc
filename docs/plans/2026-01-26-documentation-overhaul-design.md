# Documentation Overhaul Design

## Overview

Comprehensive documentation update for jax-frc targeting both researchers/physicists and developers. Uses integrated model docs (physics + implementation together) with applied-style equations (key equations with parameter guidance, not full derivations).

## Goals

1. Update model/physics docs to reflect current code state
2. Create developer/contributor documentation (currently missing)
3. Sync existing docs with recent changes (burn, transport, comparisons modules)

## Document Structure

```
jax-frc/
├── CONTRIBUTING.md                    # NEW: Contribution guidelines
├── README.md                          # UPDATE: Sync features
├── docs/
│   ├── index.md                       # UPDATE: Add new sections
│   ├── getting-started.md             # KEEP: Already good
│   ├── models/
│   │   ├── index.md                   # UPDATE: Minor tweaks
│   │   ├── resistive-mhd.md           # UPDATE: Add implementation section
│   │   ├── extended-mhd.md            # UPDATE: Add implementation section
│   │   ├── hybrid-kinetic.md          # UPDATE: Add implementation section
│   │   ├── neutral-fluid.md           # UPDATE: Add implementation section
│   │   └── burning-plasma.md          # UPDATE: Full rewrite with template
│   ├── modules/                       # NEW DIRECTORY
│   │   ├── burn.md                    # NEW: Fusion burn physics
│   │   ├── transport.md               # NEW: Anomalous transport
│   │   └── comparisons.md             # NEW: Literature validation
│   ├── developer/                     # NEW DIRECTORY
│   │   ├── architecture.md            # NEW: System overview
│   │   ├── jax-patterns.md            # NEW: JAX idioms guide
│   │   └── adding-models.md           # NEW: Tutorial
│   └── testing/
│       └── index.md                   # UPDATE: Add new test files
```

## Model Doc Template

Each model doc follows this structure to integrate physics and implementation:

```markdown
# [Model Name]

Brief description and primary use case.

## Physics Overview
- What physical regime this models (collisional, collisionless, etc.)
- Key assumptions and simplifications
- When to use this vs other models

## Governing Equations
Key equations in LaTeX with explanations:
- What each term represents physically
- Dimensionless parameters and their meaning
- NOT full derivations - link to papers for that

## Implementation
- Class structure and main entry points
- How equations map to code (e.g., "Ohm's law → `compute_electric_field()`")
- JAX-specific patterns used (scan, cond, etc.)

## Parameters
Table of configurable parameters:
| Parameter | Physical meaning | Typical range | Default |
Each with guidance on how changing it affects results.

## Solver Compatibility
Which solvers work, recommended solver, why.

## Validation
- What tests exist for this model
- Known limitations and failure modes
- Comparison to literature/other codes

## Example Usage
Minimal working code snippet.
```

## New Documents

### CONTRIBUTING.md (~300 lines)

Root-level contribution guide:
- Development setup (clone, install deps, run tests)
- Code style: Black formatting, type hints expected
- PR workflow: branch naming, commit messages, review process
- Testing requirements: what tests to add for different change types
- Link to architecture.md for deeper understanding

### docs/developer/architecture.md (~400 lines)

System overview for contributors:
- High-level diagram: Simulation orchestrates Model + Solver + Geometry
- Module dependency graph (which modules import what)
- Data flow: State → Model.compute_rhs() → Solver.step() → new State
- Extension points: where to hook in new physics
- Key abstractions: PhysicsModel protocol, Solver base class

### docs/developer/jax-patterns.md (~250 lines)

JAX idioms specific to this codebase:
- Why `lax.scan` not Python loops (with before/after examples)
- Why `lax.cond` not if/else on traced values
- Static args: when and why to use `static_argnums`
- Debugging JIT issues: common errors and fixes
- Testing JIT code: how to test both eager and compiled paths

### docs/developer/adding-models.md (~300 lines)

Tutorial for extending the system:
- Create a minimal "toy" model from scratch
- Step-by-step: implement protocol, add tests, register in __init__
- Checklist: what a complete model implementation needs

### docs/modules/burn.md (~150 lines)

Fusion burn physics module:
- Physics: D-T/D-D/D-He3 reaction rates, alpha heating, energy recovery
- Key equations: reactivity curves, power balance, Q-factor calculation
- Implementation: `burn/species.py`, `burn/physics.py`, `burn/conversion.py`
- Parameters: fuel mix ratios, conversion efficiency, alpha confinement
- Integration: how burn module couples to BurningPlasmaModel

### docs/modules/transport.md (~150 lines)

Anomalous transport module:
- Physics: turbulent transport beyond classical resistivity
- Key equations: anomalous diffusion coefficients, scaling laws
- Implementation: `transport/anomalous.py` structure
- Parameters: transport coefficients, profile dependencies
- When to enable: high-beta regimes, long confinement studies

### docs/modules/comparisons.md (~200 lines)

Literature validation framework:
- Purpose: validate against published results
- Belova et al. comparison: what's being compared, expected agreement
- Implementation: `comparisons/belova_merging.py` structure
- How to run comparisons, interpret results
- Adding new comparison cases

## Updates to Existing Documents

### README.md

- Add burn/transport/comparisons to feature list
- Update package structure diagram to show new modules
- Keep quick start example as-is (still valid)

### docs/index.md

- Add "Modules" section linking to burn, transport, comparisons docs
- Add "Developer Guide" section linking to architecture, jax-patterns, adding-models
- Update package structure tree:
  ```
  jax_frc/
  ├── burn/           # Fusion burn physics
  ├── comparisons/    # Literature validation
  ├── transport/      # Anomalous transport
  ├── validation/     # Validation infrastructure
  ...existing modules...
  ```

### docs/testing/index.md

- Add new test files to structure listing
- Mention validation tests, comparison tests
- Update test categories

### Existing Model Docs

For resistive-mhd.md, extended-mhd.md, hybrid-kinetic.md, neutral-fluid.md:
- Add "Implementation" section following template
- Add "Parameters" table with tuning guidance
- Add "Validation" section noting test coverage
- Keep existing physics content, expand structure

## Implementation Order

### Phase 1: Foundation
1. `docs/developer/architecture.md` - Need this first to reference elsewhere
2. `docs/developer/jax-patterns.md` - Core patterns referenced by model docs
3. `CONTRIBUTING.md` - Sets expectations for contributors

### Phase 2: Model Docs
4. Update `docs/models/resistive-mhd.md` - Simplest model, establish pattern
5. Update remaining model docs following same pattern
6. Full rewrite of `docs/models/burning-plasma.md` if needed

### Phase 3: New Modules
7. `docs/modules/burn.md`
8. `docs/modules/transport.md`
9. `docs/modules/comparisons.md`

### Phase 4: Integration
10. `docs/developer/adding-models.md` - Tutorial tying it together
11. Update `docs/index.md` - Add all new sections
12. Update `README.md` - Sync features
13. Update `docs/testing/index.md` - Reflect current tests

## Estimated Scope

- ~2500-3000 lines of documentation
- 15 files (8 new, 7 updates)
- New directories: `docs/developer/`, `docs/modules/`

## Design Decisions

1. **Integrated model docs** - Physics and implementation in same file rather than separate directories. Rationale: users often need both; reduces navigation.

2. **Applied equation style** - Key equations with parameter explanations, not full derivations. Rationale: researchers can find derivations in papers; what they need is "how do I use this?"

3. **Model-centric organization** - Models as primary entry points rather than workflow-based. Rationale: matches how researchers think about the problem.

4. **Separate developer guide** - Cross-cutting concerns (JAX patterns, architecture) in dedicated section rather than scattered. Rationale: avoids duplication, easier to maintain.
