# Frozen Flux Circular Advection Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update frozen flux validation to use circular advection of a magnetic loop (rigid body rotation benchmark)

**Architecture:** Replace uniform field test with localized magnetic loop that rotates back to initial position after one period

**Tech Stack:** JAX, jax-frc configuration/validation framework

---

## Design Summary

### Domain and Grid
- **Grid:** `nx=64, ny=64, nz=1` (pseudo-2D in x-y plane)
- **Domain:** x,y ∈ [-1, 1], z ∈ [-0.1, 0.1]
- **Boundaries:** Periodic on all sides

### Initial Magnetic Field
Vector potential with compact support ensures div(B) = 0:
```
A_z = A_0 * (1 - r²/R²)² for r < R, else 0
where r = sqrt((x - x0)² + (y - y0)²)

B_x = ∂A_z/∂y
B_y = -∂A_z/∂x
B_z = 0
```

**Parameters:**
- Loop center: `(x0, y0) = (0.5, 0)`
- Loop radius: `R = 0.3`
- Amplitude: `A_0 = 1.0`

### Velocity Field (Rigid Body Rotation)
```
v_x = -ω * y
v_y = +ω * x
v_z = 0
```

**Parameters:**
- Angular velocity: `ω = 2π` rad/s
- Period: `T = 1.0` s

### Runtime
- Simulation time: `t_end = 1.0` s (one full rotation)
- Timestep: `dt = 0.001` s (1000 steps)

### Validation Metrics
- L2 error between final and initial B field: threshold < 1%
- Peak amplitude ratio: threshold > 95%

---

## Implementation Tasks

### Task 1: Update FrozenFluxConfiguration

**Files:**
- Modify: `jax_frc/configurations/frozen_flux.py`

**Step 1:** Update grid parameters
```python
# Old
nx: int = 64
ny: int = 1
nz: int = 64

# New
nx: int = 64
ny: int = 64
nz: int = 1
```

**Step 2:** Update domain parameters
```python
# Old
r_min: float = 0.2
r_max: float = 1.0
z_extent: float = 0.5

# New
domain_extent: float = 1.0  # x,y ∈ [-extent, extent]
z_extent: float = 0.1       # thin z dimension
```

**Step 3:** Add magnetic loop parameters
```python
# New parameters
loop_x0: float = 0.5        # Loop center x
loop_y0: float = 0.0        # Loop center y
loop_radius: float = 0.3    # Loop radius R
loop_amplitude: float = 1.0 # Vector potential amplitude A_0
omega: float = 2 * jnp.pi   # Angular velocity [rad/s]
```

**Step 4:** Update build_geometry() for symmetric domain
```python
def build_geometry(self) -> Geometry:
    return Geometry(
        nx=self.nx, ny=self.ny, nz=self.nz,
        x_min=-self.domain_extent, x_max=self.domain_extent,
        y_min=-self.domain_extent, y_max=self.domain_extent,
        z_min=-self.z_extent, z_max=self.z_extent,
        bc_x="periodic", bc_y="periodic", bc_z="periodic",
    )
```

**Step 5:** Update build_initial_state() with magnetic loop and rotation velocity
```python
def build_initial_state(self, geometry: Geometry) -> State:
    x = geometry.x_grid
    y = geometry.y_grid

    # Distance from loop center
    r = jnp.sqrt((x - self.loop_x0)**2 + (y - self.loop_y0)**2)

    # Vector potential A_z (compact support)
    inside = r < self.loop_radius
    A_z = jnp.where(
        inside,
        self.loop_amplitude * (1 - (r / self.loop_radius)**2)**2,
        0.0
    )

    # B = curl(A_z z_hat) using finite differences
    # B_x = dA_z/dy, B_y = -dA_z/dx
    dA_dy = (jnp.roll(A_z, -1, axis=1) - jnp.roll(A_z, 1, axis=1)) / (2 * geometry.dy)
    dA_dx = (jnp.roll(A_z, -1, axis=0) - jnp.roll(A_z, 1, axis=0)) / (2 * geometry.dx)

    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B = B.at[..., 0].set(dA_dy)   # B_x = dA_z/dy
    B = B.at[..., 1].set(-dA_dx)  # B_y = -dA_z/dx

    # Rigid body rotation velocity
    v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    v = v.at[..., 0].set(-self.omega * y)  # v_x = -ω*y
    v = v.at[..., 1].set(self.omega * x)   # v_y = ω*x

    return State(B=B, E=jnp.zeros_like(B), n=..., p=..., v=v)
```

**Step 6:** Update default_runtime() and analytic_solution()

**Step 7:** Run configuration test
```bash
py -c "from jax_frc.configurations import FrozenFluxConfiguration; c = FrozenFluxConfiguration(); print(c)"
```

---

### Task 2: Update Validation Script

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1:** Update metrics to compare final vs initial B field

**Step 2:** Add peak amplitude ratio metric

**Step 3:** Update acceptance criteria
```python
ACCEPTANCE = {
    "l2_error": 0.01,           # 1% L2 error threshold
    "peak_amplitude_ratio": 0.95,  # 95% amplitude preservation
}
```

**Step 4:** Run validation
```bash
py validation/cases/analytic/frozen_flux.py
```

---

### Task 3: Update Notebook

**Files:**
- Modify: `notebooks/frozen_flux.ipynb`

**Step 1:** Update physics background section for circular advection

**Step 2:** Update configuration parameters

**Step 3:** Update visualization (2D contour plots instead of 1D line plots)

**Step 4:** Run notebook cells to verify

---

### Task 4: Verify and Commit

**Step 1:** Run full validation suite
```bash
py -m pytest tests/ -k frozen -v
```

**Step 2:** Run frozen flux validation
```bash
py validation/cases/analytic/frozen_flux.py
```

**Step 3:** Commit changes
```bash
git add jax_frc/configurations/frozen_flux.py validation/cases/analytic/frozen_flux.py notebooks/frozen_flux.ipynb
git commit -m "feat: update frozen flux to circular advection benchmark"
```
