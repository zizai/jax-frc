# Boundary Conditions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add non-periodic boundary condition support to curl/divergence operators and CT boundary handling, then validate magnetic diffusion for open and Dirichlet boundaries.

**Architecture:** Extend finite-difference operators with ghost-cell padding for Dirichlet/Neumann BCs. Apply BCs to E at CT boundaries before curl. Update magnetic diffusion configuration and add Dirichlet analytic validation case plus operator tests.

**Tech Stack:** Python, JAX, pytest.

### Task 1: Add BC-aware padding helpers in operators

**Files:**
- Modify: `jax_frc/operators.py`
- Test: `tests/test_operators_3d.py`

**Step 1: Write the failing test**

Add tests asserting curl/divergence respect Dirichlet/Neumann boundaries.

```python
def test_divergence_neumann_zero_gradient():
    geom = Geometry(nx=8, ny=8, nz=8, bc_x="neumann", bc_y="neumann", bc_z="neumann")
    F = jnp.zeros((8, 8, 8, 3))
    F = F.at[..., 0].set(1.0)  # constant field
    div_F = divergence_3d(F, geom)
    assert jnp.allclose(div_F, 0.0, atol=1e-6)

def test_curl_dirichlet_zero_boundary():
    geom = Geometry(nx=8, ny=8, nz=8, bc_x="dirichlet", bc_y="dirichlet", bc_z="dirichlet")
    F = jnp.zeros((8, 8, 8, 3))
    F = F.at[..., 1].set(geom.x_grid)  # non-periodic
    curl_F = curl_3d(F, geom)
    # Interior should be near analytic curl; boundaries should stay finite
    assert jnp.all(jnp.isfinite(curl_F))
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestCurl3D::test_curl_dirichlet_zero_boundary tests/test_operators_3d.py::TestDivergence3D::test_divergence_neumann_zero_gradient -v`  
Expected: FAIL because curl/divergence ignore non-periodic BCs.

**Step 3: Write minimal implementation**

Implement `_reflect_pad` and `_pad_with_bc` and use them inside `curl_3d` and `divergence_3d`.

```python
def _reflect_pad(f: Array, axis: int, negate: bool) -> Array:
    left = jnp.take(f, indices=0, axis=axis)
    right = jnp.take(f, indices=-1, axis=axis)
    if negate:
        left = -left
        right = -right
    return jnp.concatenate([jnp.expand_dims(left, axis), f, jnp.expand_dims(right, axis)], axis=axis)

def _pad_with_bc(f: Array, axis: int, bc: str) -> Array:
    if bc == "periodic":
        return jnp.concatenate([jnp.take(f, [-1], axis), f, jnp.take(f, [0], axis)], axis=axis)
    if bc == "dirichlet":
        return _reflect_pad(f, axis, negate=True)
    if bc == "neumann":
        return _reflect_pad(f, axis, negate=False)
    raise ValueError(...)
```

Then in `curl_3d` / `divergence_3d`, pad per axis, compute centered differences on padded arrays, and slice back to original shape.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestCurl3D::test_curl_dirichlet_zero_boundary tests/test_operators_3d.py::TestDivergence3D::test_divergence_neumann_zero_gradient -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_operators_3d.py jax_frc/operators.py
git commit -m "feat: add bc-aware curl/divergence padding"
```

### Task 2: Integrate BC handling into constrained transport solver

**Files:**
- Modify: `jax_frc/solvers/constrained_transport.py`
- Test: `tests/test_operators_3d.py` or new test in `tests/test_constrained_transport.py`

**Step 1: Write the failing test**

Create a test that E-field boundaries are handled for non-periodic BCs and the update remains finite.

```python
def test_ct_boundary_e_field_neumann():
    geom = Geometry(nx=8, ny=8, nz=8, bc_x="neumann", bc_y="neumann", bc_z="neumann")
    B = jnp.zeros((8, 8, 8, 3))
    v = jnp.zeros_like(B)
    dB = induction_rhs_ct(v, B, geom)
    assert jnp.all(jnp.isfinite(dB))
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_constrained_transport.py::test_ct_boundary_e_field_neumann -v`  
Expected: FAIL if boundaries not handled.

**Step 3: Write minimal implementation**

Implement `_extrapolate_boundary` and `_apply_boundary_E` and call it before curl in CT step.

```python
def _extrapolate_boundary(E: Array, axis: int) -> Array:
    left = jnp.take(E, indices=1, axis=axis)
    right = jnp.take(E, indices=-2, axis=axis)
    E = E.at[(slice(None),)*axis + (0,)].set(left)
    E = E.at[(slice(None),)*axis + (-1,)].set(right)
    return E

def _apply_boundary_E(E: Array, geometry: Geometry) -> Array:
    for axis, bc in enumerate([geometry.bc_x, geometry.bc_y, geometry.bc_z]):
        if bc != "periodic":
            E = _extrapolate_boundary(E, axis)
    return E
```

Integrate in CT step (before curl).

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_constrained_transport.py::test_ct_boundary_e_field_neumann -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add jax_frc/solvers/constrained_transport.py tests/test_constrained_transport.py
git commit -m "feat: apply E-field BCs in CT solver"
```

### Task 3: Update magnetic diffusion configuration for open boundaries

**Files:**
- Modify: `jax_frc/configurations/magnetic_diffusion.py`
- Modify: `validation/cases/analytic/magnetic_diffusion.py`
- Test: `tests/test_validation_cases.py` (or new focused test)

**Step 1: Write the failing test**

```python
def test_magnetic_diffusion_open_bc_defaults():
    config = MagneticDiffusionConfiguration()
    assert config.bc_x == "neumann"
    assert config.bc_y == "neumann"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_cases.py::test_magnetic_diffusion_open_bc_defaults -v`  
Expected: FAIL (defaults still periodic).

**Step 3: Write minimal implementation**

Update defaults to Neumann and adjust any configuration wiring for validation case.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_cases.py::test_magnetic_diffusion_open_bc_defaults -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add jax_frc/configurations/magnetic_diffusion.py validation/cases/analytic/magnetic_diffusion.py tests/test_validation_cases.py
git commit -m "feat: use open boundary defaults for magnetic diffusion"
```

### Task 4: Add Dirichlet magnetic diffusion validation case

**Files:**
- Create: `validation/cases/analytic/magnetic_diffusion_dirichlet.py`
- Test: `tests/test_validation_cases.py` (or new test file)

**Step 1: Write the failing test**

```python
def test_magnetic_diffusion_dirichlet_case_exists():
    from validation.cases.analytic.magnetic_diffusion_dirichlet import run_validation
    assert callable(run_validation)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_cases.py::test_magnetic_diffusion_dirichlet_case_exists -v`  
Expected: FAIL (file missing).

**Step 3: Write minimal implementation**

Implement Dirichlet analytic solution using Fourier series modes and run validation to compute L2 error and boundary enforcement.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_cases.py::test_magnetic_diffusion_dirichlet_case_exists -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add validation/cases/analytic/magnetic_diffusion_dirichlet.py tests/test_validation_cases.py
git commit -m "feat: add dirichlet magnetic diffusion validation"
```

### Task 5: Full verification

**Files:**
- None (test only)

**Step 1: Run full test suite (fast)**

Run: `py -m pytest tests/ -k "not slow"`  
Expected: PASS.

**Step 2: Run validation suite**

Run: `py -m scripts/run_validation.py --all`  
Expected: PASS.

**Step 3: Commit**

```bash
git commit --allow-empty -m "test: verify boundary condition updates"
```

Plan complete and saved to `docs/plans/2026-01-28-boundary-conditions-implementation.md`. Two execution options:

1. Subagent-Driven (this session)
2. Parallel Session (separate)

Which approach?
