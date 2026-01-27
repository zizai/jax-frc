# CLAUDE.md

JAX-based plasma physics simulation for FRC research. GPU-accelerated, uses JAX's functional patterns throughout.

## Commands

```bash
py -m pytest tests/ -v                 # Run all tests
py -m pytest tests/ -k "not slow"      # Skip slow physics tests
py -m scripts/run_validation.py --all  # Run validation tests
```

## JAX Patterns

- **Loops**: Use `lax.scan`, not Python for-loops (breaks JIT tracing)
- **Branches**: Use `lax.cond`, not if/else on traced values
- **State**: Immutable tuples/dataclasses passed through `step()` functions
- **JIT**: All compute functions `@jax.jit`; use `static_argnums` for config/shape args

## Testing Patterns

- **Fast tests**: Mock physics operators, test logic separately from numerics
- **Slow tests**: Mark with `@pytest.mark.slow` if they run actual simulations
- **Invariants**: Test conservation laws, symmetries, boundary conditions

## Validation Patterns

- **Visualization**: Use plots, reports and notebooks for each validation case
- **Major changes**: Run validation tests before pushing major changes upstream (models, solvers, or overall architecture)

## Gotchas

- **Static args**: Forgetting `static_argnums` causes recompilation or tracing errors
- **Magic constants**: Use `jax_frc.constants`, not hardcoded numbers
- **Slow tests**: Don't put physics in unit tests; use small grids or mocks
- **Numerical instabilities**: Check against invalid outputs from solvers
- **Notational inconsistencies**: Make sure variables keep the same units and convention thoughout its lifetime
