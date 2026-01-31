# Test Runtime Optimization Design

**Date:** 2026-01-31  
**Goal:** Reduce `tests/` (excluding slow tests) runtime to under 5 minutes while preserving core invariants, and keep `validation/` as a separate concern.

## Scope and Success Criteria

- Optimize only `tests/` (not `validation/`).
- It is acceptable to reduce coverage scope (smaller grids, fewer steps, fewer parameter sweeps).
- Not-slow suite should complete in under 5 minutes on this machine.
- Preserve key invariants: conservation, symmetry, boundedness, and basic configuration correctness.

## Recommended Approach

Use a measurement-driven loop with small, 1-minute optimization chunks:

1. Run `pytest --durations=20` on the not-slow suite to identify the worst offenders.
2. Target a small set of heavy tests per chunk and reduce computational scope:
   - Smaller grid sizes (e.g., 128 -> 64)
   - Fewer time steps or iterations
   - Narrower parameter sweeps (1-2 representative values)
   - Prefer mock boundaries for tests that only validate control flow
3. Re-run a focused subset to validate the change.
4. Re-check durations to verify progress.

This approach minimizes risk by making incremental, reversible changes while keeping critical checks intact.

## Alternatives Considered

- **Mock or patch more physics operators:** Faster but can miss integration issues.
- **Move heavy tests to a new marker/folder:** Changes the meaning of not-slow; not recommended unless policy changes are desired.
- **Global JIT/precision toggles:** Risky because they can mask JIT-specific issues.

## Audit of Slow Functions (Separate Doc)

Create a standalone audit file at:
`docs/audits/2026-01-31-test-runtime-audit.md`

The audit should include:
- Command + environment used for timings
- Top-N slow tests with wall times
- Suspected slow functions/modules for each test
- Next-action notes for future improvements

This document will be updated as we make incremental improvements.

## Risks and Mitigations

- **Risk:** Reduced coverage misses edge cases.  
  **Mitigation:** Preserve invariants and a minimal representative parameter set.
- **Risk:** Over-optimization hides real regressions.  
  **Mitigation:** Maintain a clear separation of unit vs. validation workflows.
