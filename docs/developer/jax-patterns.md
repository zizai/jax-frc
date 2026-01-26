# JAX Patterns in jax-frc

This guide covers JAX-specific patterns used throughout the codebase. Understanding these is essential for contributing.

## Why JAX?

JAX provides:
- **Automatic differentiation** for gradient-based optimization
- **JIT compilation** for GPU/TPU acceleration
- **Functional transformations** (vmap, pmap) for vectorization

The tradeoff: code must follow functional patterns that enable tracing.

## Core Patterns

### Use `lax.scan` Instead of Python Loops

**Problem**: Python for-loops break JIT tracing.

```python
# BAD - breaks JIT
def time_loop(state, n_steps):
    for _ in range(n_steps):
        state = step(state)
    return state

# GOOD - JIT-compatible
def time_loop(state, n_steps):
    def body(state, _):
        return step(state), None
    final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
    return final_state
```

### Use `lax.cond` Instead of if/else on Traced Values

**Problem**: Python if/else on JAX arrays causes tracing errors.

```python
# BAD - traced value in condition
def apply_bc(psi, use_dirichlet):
    if use_dirichlet:  # Error if use_dirichlet is traced
        return psi.at[0].set(0.0)
    return psi

# GOOD - lax.cond for traced conditions
def apply_bc(psi, use_dirichlet):
    return jax.lax.cond(
        use_dirichlet,
        lambda p: p.at[0].set(0.0),
        lambda p: p,
        psi
    )
```

**When Python if/else IS okay**: Config values known at compile time (use `static_argnums`).

### Static Arguments for JIT

Use `static_argnums` for arguments that:
- Affect array shapes
- Are used in Python control flow
- Are configuration objects

```python
@partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    # geometry.nr, geometry.nz used for shapes - must be static
    ...
```

**Forgetting static_argnums causes**:
- Recompilation on every call (slow)
- Or ConcretizationError (fails)

### Immutable State Updates

JAX arrays are immutable. Use `.at[].set()` for updates:

```python
# BAD - mutation doesn't work
psi[0, :] = 0.0

# GOOD - returns new array
psi = psi.at[0, :].set(0.0)
```

## Debugging JIT Issues

### ConcretizationError

**Symptom**: "Abstract tracer value encountered where concrete value is expected"

**Cause**: Using traced value where Python needs concrete value (if/else, shape, indexing).

**Fix**:
- Use `lax.cond` instead of if/else
- Add argument to `static_argnums`
- Use `jax.debug.print` for debugging

### Shape Errors

**Symptom**: "Shapes must be 1D sequences of concrete values"

**Cause**: Array shape depends on traced value.

**Fix**: Ensure shapes are determined by static arguments only.

### Slow Recompilation

**Symptom**: First call fast, subsequent calls slow.

**Cause**: JIT cache miss due to changing "static" arguments.

**Fix**: Ensure config objects are truly static (same Python object identity).

## Testing JIT Code

Test both eager and compiled paths:

```python
def test_compute_rhs_eager():
    """Test without JIT for easier debugging."""
    with jax.disable_jit():
        result = model.compute_rhs(state, geometry)
    assert result.psi.shape == state.psi.shape

def test_compute_rhs_jit():
    """Test with JIT to catch tracing issues."""
    result = model.compute_rhs(state, geometry)  # JIT enabled
    assert result.psi.shape == state.psi.shape
```

## Common Gotchas

| Issue | Symptom | Solution |
|-------|---------|----------|
| Python loop in JIT | Slow or ConcretizationError | Use `lax.scan` |
| if/else on traced value | ConcretizationError | Use `lax.cond` |
| Missing static_argnums | Recompilation or error | Add to decorator |
| Mutable update | No effect | Use `.at[].set()` |
| Print in JIT | No output | Use `jax.debug.print` |

## Further Reading

- [JAX Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JIT Compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- [Stateful Computations](https://jax.readthedocs.io/en/latest/jax-101/07-state.html)
