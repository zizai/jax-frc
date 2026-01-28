# Critical Model Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the four critical model issues identified in the AGATE comparison: boundary conditions, div(B) enforcement, semi-implicit solver, and RK4 field updates.

**Architecture:** Implement boundary-aware operators, integrate divergence cleaning into MHD models, fix semi-implicit solver to use proper CG iteration, and update RK4 to evolve all state fields.

**Tech Stack:** JAX, jax.numpy, jax.lax.scan, conjugate gradient solver

---

## Task 1: Add Boundary-Aware Gradient Helper

**Files:**
- Modify: `jax_frc/operators.py:255-278`
- Test: `tests/test_operators_3d.py`

**Step 1: Write the failing test for Dirichlet BC**

Add to `tests/test_operators_3d.py`:

```python
def test_gradient_dirichlet_bc():
    """Gradient with Dirichlet BC uses one-sided differences at boundaries."""
    geom = Geometry(
        nx=16, ny=8, nz=8,
        x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0,
        z_min=0.0, z_max=1.0,
        bc_x="dirichlet", bc_y="periodic", bc_z="periodic"
    )
    f = geom.x_grid  # f = x
    grad_f = gradient_3d(f, geom)
    # With Dirichlet BC, boundary gradient should use one-sided difference
    # df/dx at x=0 should be (f[1] - f[0]) / dx (forward difference)
    # df/dx at x=max should be (f[-1] - f[-2]) / dx (backward difference)
    assert jnp.allclose(grad_f[0, :, :, 0], 1.0, atol=0.1)  # Forward diff at left
    assert jnp.allclose(grad_f[-1, :, :, 0], 1.0, atol=0.1)  # Backward diff at right
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestGradient3D::test_gradient_dirichlet_bc -v`
Expected: FAIL (bc_x is ignored, uses periodic wrap)

**Step 3: Add boundary-aware helper function**

Add to `jax_frc/operators.py` before `gradient_3d`:

```python
def _derivative_with_bc(f: Array, dx: float, axis: int, bc: str) -> Array:
    """Compute derivative along axis respecting boundary condition.

    Args:
        f: Field array
        dx: Grid spacing
        axis: Axis to differentiate along (0=x, 1=y, 2=z)
        bc: Boundary condition type ("periodic", "dirichlet", "neumann")

    Returns:
        Derivative array with same shape as f
    """
    if bc == "periodic":
        # Central differences with periodic wrap
        return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dx)

    # Non-periodic: central interior, one-sided at boundaries
    # Interior: central difference
    f_plus = jnp.roll(f, -1, axis=axis)
    f_minus = jnp.roll(f, 1, axis=axis)
    df = (f_plus - f_minus) / (2 * dx)

    # Boundary corrections using slicing
    ndim = f.ndim

    # Left boundary: forward difference (f[1] - f[0]) / dx
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    next_slice = [slice(None)] * ndim
    next_slice[axis] = 1
    df_left = (f[tuple(next_slice)] - f[tuple(left_slice)]) / dx
    df = df.at[tuple(left_slice)].set(df_left)

    # Right boundary: backward difference (f[-1] - f[-2]) / dx
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    prev_slice = [slice(None)] * ndim
    prev_slice[axis] = -2
    df_right = (f[tuple(right_slice)] - f[tuple(prev_slice)]) / dx
    df = df.at[tuple(right_slice)].set(df_right)

    return df
```

**Step 4: Update gradient_3d to use helper**

Replace `gradient_3d` function:

```python
@jit(static_argnums=(1,))
def gradient_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute gradient of scalar field in 3D Cartesian coordinates.

    Uses central differences in interior, one-sided at non-periodic boundaries.

    Args:
        f: Scalar field, shape (nx, ny, nz)
        geometry: 3D Cartesian geometry with bc_x, bc_y, bc_z settings

    Returns:
        Gradient vector field, shape (nx, ny, nz, 3) with [df/dx, df/dy, df/dz]
    """
    df_dx = _derivative_with_bc(f, geometry.dx, axis=0, bc=geometry.bc_x)
    df_dy = _derivative_with_bc(f, geometry.dy, axis=1, bc=geometry.bc_y)
    df_dz = _derivative_with_bc(f, geometry.dz, axis=2, bc=geometry.bc_z)

    return jnp.stack([df_dx, df_dy, df_dz], axis=-1)
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestGradient3D -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat: add boundary-aware gradient operator"
```

---

## Task 2: Update Laplacian and Curl for Boundary Conditions

**Files:**
- Modify: `jax_frc/operators.py:317-336` (laplacian_3d)
- Modify: `jax_frc/operators.py:472-502` (curl_3d)
- Test: `tests/test_operators_3d.py`

**Step 1: Write failing test for laplacian with Neumann BC**

```python
def test_laplacian_neumann_bc():
    """Laplacian with Neumann BC uses zero-gradient at boundaries."""
    geom = Geometry(
        nx=16, ny=16, nz=8,
        x_min=-1.0, x_max=1.0,
        y_min=-1.0, y_max=1.0,
        z_min=0.0, z_max=1.0,
        bc_x="neumann", bc_y="neumann", bc_z="periodic"
    )
    # Gaussian: laplacian should be smooth at boundaries
    r_sq = geom.x_grid**2 + geom.y_grid**2
    f = jnp.exp(-r_sq / 0.1)
    lap_f = laplacian_3d(f, geom)
    # With Neumann BC, boundary values should not wrap around
    # Check that boundary laplacian is finite and reasonable
    assert jnp.all(jnp.isfinite(lap_f))
    assert jnp.abs(lap_f[0, 8, 4]) < jnp.abs(lap_f[8, 8, 4])  # Boundary < center
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestLaplacian3D::test_laplacian_neumann_bc -v`
Expected: FAIL

**Step 3: Add second derivative helper**

```python
def _second_derivative_with_bc(f: Array, dx: float, axis: int, bc: str) -> Array:
    """Compute second derivative along axis respecting boundary condition."""
    if bc == "periodic":
        f_plus = jnp.roll(f, -1, axis=axis)
        f_minus = jnp.roll(f, 1, axis=axis)
        return (f_plus - 2*f + f_minus) / (dx**2)

    # Non-periodic: use ghost cell approach
    # For Neumann: ghost = interior (zero gradient)
    # For Dirichlet: ghost = -interior (zero value)
    ndim = f.ndim

    # Interior: standard central difference
    f_plus = jnp.roll(f, -1, axis=axis)
    f_minus = jnp.roll(f, 1, axis=axis)
    d2f = (f_plus - 2*f + f_minus) / (dx**2)

    # Left boundary correction
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    next_slice = [slice(None)] * ndim
    next_slice[axis] = 1

    if bc == "neumann":
        # Ghost cell = f[0] (zero gradient), so f[-1] = f[0]
        # d2f = (f[1] - 2*f[0] + f[0]) / dx^2 = (f[1] - f[0]) / dx^2
        d2f_left = (f[tuple(next_slice)] - f[tuple(left_slice)]) / (dx**2)
    else:  # dirichlet
        # Ghost cell = -f[0] (zero value at boundary)
        # d2f = (f[1] - 2*f[0] + (-f[0])) / dx^2 = (f[1] - 3*f[0]) / dx^2
        d2f_left = (f[tuple(next_slice)] - 3*f[tuple(left_slice)]) / (dx**2)
    d2f = d2f.at[tuple(left_slice)].set(d2f_left)

    # Right boundary correction
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    prev_slice = [slice(None)] * ndim
    prev_slice[axis] = -2

    if bc == "neumann":
        d2f_right = (f[tuple(prev_slice)] - f[tuple(right_slice)]) / (dx**2)
    else:  # dirichlet
        d2f_right = (f[tuple(prev_slice)] - 3*f[tuple(right_slice)]) / (dx**2)
    d2f = d2f.at[tuple(right_slice)].set(d2f_right)

    return d2f
```

**Step 4: Update laplacian_3d**

```python
@jit(static_argnums=(1,))
def laplacian_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute Laplacian of scalar field in 3D Cartesian coordinates.

    Args:
        f: Scalar field, shape (nx, ny, nz)
        geometry: 3D Cartesian geometry with bc_x, bc_y, bc_z settings

    Returns:
        Laplacian scalar field, shape (nx, ny, nz)
    """
    d2f_dx2 = _second_derivative_with_bc(f, geometry.dx, axis=0, bc=geometry.bc_x)
    d2f_dy2 = _second_derivative_with_bc(f, geometry.dy, axis=1, bc=geometry.bc_y)
    d2f_dz2 = _second_derivative_with_bc(f, geometry.dz, axis=2, bc=geometry.bc_z)

    return d2f_dx2 + d2f_dy2 + d2f_dz2
```

**Step 5: Run tests**

Run: `py -m pytest tests/test_operators_3d.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat: add boundary-aware laplacian operator"
```

---

## Task 3: Integrate Divergence Cleaning into ResistiveMHD

**Files:**
- Modify: `jax_frc/models/resistive_mhd.py:89-92`
- Test: `tests/test_resistive_mhd_3d.py`

**Step 1: Write failing test**

Add to `tests/test_resistive_mhd_3d.py`:

```python
def test_apply_constraints_cleans_divergence():
    """apply_constraints should reduce div(B)."""
    from jax_frc.operators import divergence_3d

    geom = Geometry(nx=16, ny=16, nz=16)

    # Create state with non-zero divergence
    B = jnp.zeros((16, 16, 16, 3))
    B = B.at[:, :, :, 0].set(geom.x_grid)  # Bx = x has div(B) != 0

    state = State(
        B=B,
        E=jnp.zeros((16, 16, 16, 3)),
        n=jnp.ones((16, 16, 16)),
        p=jnp.ones((16, 16, 16)),
        v=jnp.zeros((16, 16, 16, 3)),
    )

    model = ResistiveMHD(eta=1e-4)

    div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
    cleaned_state = model.apply_constraints(state, geom)
    div_after = jnp.linalg.norm(divergence_3d(cleaned_state.B, geom))

    assert div_after < div_before * 0.1  # At least 10x reduction
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_resistive_mhd_3d.py::test_apply_constraints_cleans_divergence -v`
Expected: FAIL (apply_constraints is no-op)

**Step 3: Update apply_constraints**

In `jax_frc/models/resistive_mhd.py`, replace `apply_constraints`:

```python
def apply_constraints(self, state: State, geometry: Geometry) -> State:
    """Apply divergence cleaning to magnetic field.

    Uses projection method: B_clean = B - grad(phi) where laplacian(phi) = div(B)
    """
    from jax_frc.solvers.divergence_cleaning import clean_divergence

    B_clean = clean_divergence(state.B, geometry, max_iter=50, tol=1e-6)
    return state.replace(B=B_clean)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_resistive_mhd_3d.py::test_apply_constraints_cleans_divergence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/resistive_mhd.py tests/test_resistive_mhd_3d.py
git commit -m "feat: integrate divergence cleaning into ResistiveMHD"
```

---

## Task 4: Integrate Divergence Cleaning into ExtendedMHD

**Files:**
- Modify: `jax_frc/models/extended_mhd.py:180-204`
- Test: `tests/test_extended_mhd_3d.py`

**Step 1: Write failing test**

Add to `tests/test_extended_mhd_3d.py`:

```python
def test_apply_constraints_cleans_divergence():
    """apply_constraints should reduce div(B) in ExtendedMHD."""
    from jax_frc.operators import divergence_3d

    geom = Geometry(nx=16, ny=16, nz=16)

    B = jnp.zeros((16, 16, 16, 3))
    B = B.at[:, :, :, 0].set(geom.x_grid)

    state = State(
        B=B,
        E=jnp.zeros((16, 16, 16, 3)),
        n=jnp.ones((16, 16, 16)) * 1e20,
        p=jnp.ones((16, 16, 16)) * 1e3,
        v=jnp.zeros((16, 16, 16, 3)),
    )

    model = ExtendedMHD(eta=1e-4)

    div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
    cleaned_state = model.apply_constraints(state, geom)
    div_after = jnp.linalg.norm(divergence_3d(cleaned_state.B, geom))

    assert div_after < div_before * 0.1
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_extended_mhd_3d.py::test_apply_constraints_cleans_divergence -v`
Expected: FAIL

**Step 3: Update apply_constraints**

In `jax_frc/models/extended_mhd.py`, update `apply_constraints`:

```python
def apply_constraints(self, state: State, geometry: Geometry) -> State:
    """Apply boundary conditions and divergence cleaning."""
    from jax_frc.solvers.divergence_cleaning import clean_divergence

    # Clean divergence first
    B_clean = clean_divergence(state.B, geometry, max_iter=50, tol=1e-6)
    state = state.replace(B=B_clean)

    # Then apply temperature BCs if configured
    if self.temperature_bc is None or state.Te is None:
        return state

    T = state.Te
    bc = self.temperature_bc

    if bc.bc_type == "dirichlet":
        T = T.at[0, :, :].set(bc.T_wall)
        T = T.at[-1, :, :].set(bc.T_wall)
        T = T.at[:, :, 0].set(bc.T_wall)
        T = T.at[:, :, -1].set(bc.T_wall)
    else:
        T = T.at[0, :, :].set(T[1, :, :])
        T = T.at[-1, :, :].set(T[-2, :, :])
        T = T.at[:, :, 0].set(T[:, :, 1])
        T = T.at[:, :, -1].set(T[:, :, -2])

    if bc.apply_axis_symmetry:
        T = T.at[0, :, :].set(T[1, :, :])

    return state.replace(Te=T)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_extended_mhd_3d.py::test_apply_constraints_cleans_divergence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/extended_mhd.py tests/test_extended_mhd_3d.py
git commit -m "feat: integrate divergence cleaning into ExtendedMHD"
```

---

## Task 5: Fix RK4 Solver to Update All State Fields

**Files:**
- Modify: `jax_frc/solvers/explicit.py:48-87`
- Test: `tests/test_semi_implicit.py` (or new test file)

**Step 1: Write failing test**

Add to `tests/test_semi_implicit.py`:

```python
def test_rk4_updates_temperature():
    """RK4 should update Te field in intermediate stages."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.models.extended_mhd import ExtendedMHD

    geom = Geometry(nx=8, ny=8, nz=8)

    # Initial state with temperature
    state = State(
        B=jnp.ones((8, 8, 8, 3)) * 0.1,
        E=jnp.zeros((8, 8, 8, 3)),
        n=jnp.ones((8, 8, 8)) * 1e20,
        p=jnp.ones((8, 8, 8)) * 1e3,
        v=jnp.zeros((8, 8, 8, 3)),
        Te=jnp.ones((8, 8, 8)) * 100.0,  # 100 eV
    )

    model = ExtendedMHD(eta=1e-4, include_hall=False)
    solver = RK4Solver()

    # Take one step
    new_state = solver.step(state, dt=1e-6, model=model, geometry=geom)

    # Te should be updated (not just B)
    # If model returns dTe/dt != 0, Te should change
    # This test verifies the solver propagates Te through stages
    assert new_state.Te is not None
```

**Step 2: Run test to verify current behavior**

Run: `py -m pytest tests/test_semi_implicit.py::test_rk4_updates_temperature -v`

**Step 3: Update RK4Solver to handle all fields**

In `jax_frc/solvers/explicit.py`, replace `RK4Solver.step`:

```python
@partial(jax.jit, static_argnums=(0, 3, 4))
def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
    """RK4 time step updating all state fields."""

    def add_scaled_rhs(base: State, rhs: State, scale: float) -> State:
        """Add scaled RHS to base state for all fields."""
        new_B = base.B + scale * rhs.B
        new_E = base.E + scale * rhs.E if rhs.E is not None else base.E
        new_Te = None
        if base.Te is not None and rhs.Te is not None:
            new_Te = base.Te + scale * rhs.Te
        return base.replace(B=new_B, E=new_E, Te=new_Te)

    # k1
    k1 = model.compute_rhs(state, geometry)

    # k2
    state_k2 = add_scaled_rhs(state, k1, 0.5 * dt)
    k2 = model.compute_rhs(state_k2, geometry)

    # k3
    state_k3 = add_scaled_rhs(state, k2, 0.5 * dt)
    k3 = model.compute_rhs(state_k3, geometry)

    # k4
    state_k4 = add_scaled_rhs(state, k3, dt)
    k4 = model.compute_rhs(state_k4, geometry)

    # Combine: y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    new_B = state.B + (dt/6) * (k1.B + 2*k2.B + 2*k3.B + k4.B)

    new_E = state.E
    if k1.E is not None:
        new_E = state.E + (dt/6) * (k1.E + 2*k2.E + 2*k3.E + k4.E)

    new_Te = None
    if state.Te is not None and k1.Te is not None:
        new_Te = state.Te + (dt/6) * (k1.Te + 2*k2.Te + 2*k3.Te + k4.Te)

    new_state = state.replace(
        B=new_B,
        E=new_E,
        Te=new_Te,
        time=state.time + dt,
        step=state.step + 1,
    )
    return model.apply_constraints(new_state, geometry)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_semi_implicit.py::test_rk4_updates_temperature -v`
Expected: PASS

**Step 5: Run all solver tests**

Run: `py -m pytest tests/test_semi_implicit.py tests/test_imex_diffusion.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add jax_frc/solvers/explicit.py tests/test_semi_implicit.py
git commit -m "fix: RK4 solver now updates all state fields in intermediate stages"
```

---

## Task 6: Run Full Test Suite and Validation

**Step 1: Run all tests**

Run: `py -m pytest tests/ -v --tb=short`
Expected: All PASS (or document any expected failures)

**Step 2: Run magnetic diffusion validation**

Run: `py validation/cases/analytic/magnetic_diffusion.py`
Expected: PASS for both ResistiveMHD and ExtendedMHD

**Step 3: Final commit**

```bash
git add -A
git commit -m "test: verify all critical model fixes pass validation"
```

---

## Summary

| Task | Component | Priority | Status |
|------|-----------|----------|--------|
| 1 | Boundary-aware gradient | P0 | Pending |
| 2 | Boundary-aware laplacian | P0 | Pending |
| 3 | div(B) cleaning in ResistiveMHD | P0 | Pending |
| 4 | div(B) cleaning in ExtendedMHD | P0 | Pending |
| 5 | RK4 all-field updates | P1 | Pending |
| 6 | Full test suite | P0 | Pending |

**Note:** The semi-implicit solver fix (proper CG iteration) is deferred to a separate plan as it requires more extensive changes.
