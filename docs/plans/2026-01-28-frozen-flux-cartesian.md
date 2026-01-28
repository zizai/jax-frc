# Frozen Flux Cartesian Validation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor frozen flux configuration, validation, and notebook to use a Cartesian, ideal-MHD frozen-flux formulation (Rm >> 1) with default grid 64x1x64 (or 64x64x1), matching the analytic constant-B solution and the Wikipedia magnetic diffusion derivation.

**Architecture:** Keep the configuration as uniform B and uniform v in 3D Cartesian space; the solver under test advances the induction equation. The analytic reference remains constant B because curl(v x B) = 0 for uniform fields. Validation compares solver output to that analytic field; the notebook explains the Cartesian induction equation and the limiting ideal-MHD form.

**Tech Stack:** Python (jax, jax.numpy), jax_frc configuration/validation utilities, matplotlib notebook.

### Task 1: Update FrozenFluxConfiguration defaults and tests for Cartesian grid

**Files:**
- Modify: `tests/test_configurations.py`
- Modify: `jax_frc/configurations/frozen_flux.py`

**Step 1: Write the failing test**

Add a new test in `tests/test_configurations.py`:

```python
def test_frozen_flux_default_grid_dims_cartesian():
    from jax_frc.configurations import FrozenFluxConfiguration

    config = FrozenFluxConfiguration()

    assert config.nx == 64
    assert config.ny == 1
    assert config.nz == 64
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_configurations.py::test_frozen_flux_default_grid_dims_cartesian -v`
Expected: FAIL because defaults are still 64x4x8.

**Step 3: Write minimal implementation**

Update `jax_frc/configurations/frozen_flux.py`:
- Set defaults to `nx=64`, `ny=1`, `nz=64` (or `nx=64`, `ny=64`, `nz=1` if you choose that variant; keep consistent across validation and notebook).
- Update the docstring and inline comments to describe Cartesian induction equation and the Rm >> 1 limit.
- Ensure the geometry uses the thin dimension as periodic (ny=1 or nz=1) and retains non-periodic boundaries on the thick dimensions.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_configurations.py::test_frozen_flux_default_grid_dims_cartesian -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_configurations.py jax_frc/configurations/frozen_flux.py
git commit -m "test(config): enforce frozen flux Cartesian defaults"
```

### Task 2: Refactor validation case + YAML to Cartesian formulation

**Files:**
- Create: `tests/test_validation_cases.py`
- Modify: `validation/cases/analytic/frozen_flux.py`
- Modify: `validation/cases/analytic/frozen_flux.yaml`

**Step 1: Write the failing test**

Create `tests/test_validation_cases.py`:

```python
def test_frozen_flux_validation_setup_uses_cartesian_defaults():
    from validation.cases.analytic.frozen_flux import setup_configuration

    cfg = setup_configuration()
    assert cfg["nx"] == 64
    assert cfg["ny"] == 1
    assert cfg["nz"] == 64
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_cases.py::test_frozen_flux_validation_setup_uses_cartesian_defaults -v`
Expected: FAIL until validation pulls the updated defaults.

**Step 3: Write minimal implementation**

Update `validation/cases/analytic/frozen_flux.py`:
- Adjust the module docstring to cite the Cartesian induction equation and its ideal-MHD limit, noting that uniform v and B yield constant B.
- Rename B_theta/B_phi variables to B_y (or B_component) to match Cartesian axes where the field is initialized.
- Ensure plotting uses Cartesian axes (x line at y=0, z=mid) and labels match.
- Keep the L2 acceptance at 0.01.

Update `validation/cases/analytic/frozen_flux.yaml`:
- Replace cylindrical references (r, B_theta, B_phi) with Cartesian x/y/z naming.
- Update the reference formula to use `x` (or a generic coordinate) and constant B.
- Update plot axis to x and field to B_y (or equivalent).

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_cases.py::test_frozen_flux_validation_setup_uses_cartesian_defaults -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_validation_cases.py validation/cases/analytic/frozen_flux.py validation/cases/analytic/frozen_flux.yaml
git commit -m "refactor(validation): align frozen flux with Cartesian formulation"
```

### Task 3: Refactor frozen_flux notebook to Cartesian formulation

**Files:**
- Modify: `notebooks/frozen_flux.ipynb`

**Step 1: Update physics narrative and equations**

- Replace 2D cylindrical discussion with 3D Cartesian induction equation form.
- Include the diffusion term and the ideal-MHD limit (Rm >> 1).
- Explicitly state the analytic solution for uniform v and B: B(t) = constant.

**Step 2: Update configuration and setup cells**

- Replace `nr`, `r`, `z` with `nx`, `ny`, `nz`, and x/y/z grids.
- Use the FrozenFluxConfiguration defaults (64x1x64 or 64x64x1), and remove the Gaussian axial profile.
- Ensure the configuration cell runs without errors (remove the old nr/nz mismatch).

**Step 3: Update visualization and metrics**

- Replace 2D contour plots with a Cartesian slice (e.g., B_y on x-z at y=0) plus a 1D line plot along x.
- Update metric computation to compare against a uniform B analytic field and report L2 error < 0.01.
- Keep plots and metrics consistent with the validation script naming.

**Step 4: Manual notebook sanity check**

Run the notebook cells (or execute via nbconvert if available) to ensure there are no runtime errors and the figures render.

**Step 5: Commit**

```bash
git add notebooks/frozen_flux.ipynb
git commit -m "docs(notebook): refactor frozen flux to Cartesian formulation"
```
