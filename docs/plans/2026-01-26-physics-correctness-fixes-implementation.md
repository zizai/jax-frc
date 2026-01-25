# Physics Correctness Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical physics bugs in `jax_frc/models/` classes to match corrected root-level implementations

**Architecture:** The `jax_frc/models/` module contains object-oriented physics models, while root-level files (`hybrid_kinetic.py`, `extended_mhd.py`) contain functional implementations. The root files have been corrected; we synchronize the module versions and add missing collision operator and divergence cleaning.

**Tech Stack:** JAX, jax.numpy, pytest

---

## Task 1: Fix Extended MHD Cylindrical Curl Operator

The `_compute_current` method in `jax_frc/models/extended_mhd.py` incorrectly computes J_z. The correct cylindrical curl formula is:

```
J_z = (1/μ₀) * (1/r) * ∂(r·B_θ)/∂r = (1/μ₀) * (B_θ/r + ∂B_θ/∂r)
```

**Files:**
- Modify: `jax_frc/models/extended_mhd.py:111-126`
- Test: `tests/test_operators_physics.py` (create)

**Step 1: Write failing test for cylindrical curl correctness**

Create `tests/test_operators_physics.py`:

```python
"""Physics correctness tests for differential operators."""
import pytest
import jax.numpy as jnp
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.geometry import Geometry


class TestCylindricalCurl:
    """Tests for cylindrical curl operator correctness."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return Geometry(
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=32, nz=64
        )

    @pytest.fixture
    def model(self):
        """Create ExtendedMHD model."""
        resistivity = SpitzerResistivity(eta_0=1e-6)
        from jax_frc.models.extended_mhd import HaloDensityModel
        halo = HaloDensityModel()
        return ExtendedMHD(resistivity=resistivity, halo_model=halo)

    def test_jz_includes_b_theta_over_r_term(self, geometry, model):
        """J_z should include B_theta/r term, not just dB_theta/dr.

        For B_theta = r (linear in r), the correct J_z is:
            J_z = (1/mu0) * (B_theta/r + dB_theta/dr)
                = (1/mu0) * (r/r + 1)
                = (1/mu0) * 2

        The incorrect formula (just dB_theta/dr) gives:
            J_z = (1/mu0) * 1
        """
        MU0 = 1.2566e-6
        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # B_theta = r (linear profile)
        B_r = jnp.zeros((nr, nz))
        B_theta = r * jnp.ones((1, nz))  # B_theta = r
        B_z = jnp.zeros((nr, nz))

        J_r, J_phi, J_z = model._compute_current(B_r, B_theta, B_z, dr, dz, r)

        # Expected: J_z = 2/mu0 everywhere (except boundaries)
        expected_J_z = 2.0 / MU0

        # Check interior points (avoid boundaries)
        interior_J_z = J_z[5:-5, 5:-5]
        interior_expected = jnp.ones_like(interior_J_z) * expected_J_z

        # Should match within 5% (finite difference error)
        relative_error = jnp.abs(interior_J_z - interior_expected) / expected_J_z
        max_error = float(jnp.max(relative_error))

        assert max_error < 0.05, f"J_z incorrect: max relative error {max_error:.2%}"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_physics.py::TestCylindricalCurl::test_jz_includes_b_theta_over_r_term -v`

Expected: FAIL with assertion error showing ~50% relative error (because it computes J_z = 1/mu0 instead of 2/mu0)

**Step 3: Fix `_compute_current` to use correct cylindrical formula**

In `jax_frc/models/extended_mhd.py`, replace the `_compute_current` method:

```python
def _compute_current(self, B_r, B_phi, B_z, dr, dz, r):
    """Compute J = curl(B) / mu_0 in cylindrical coordinates.

    Curl in cylindrical (r, theta, z) with axisymmetry (d/dtheta = 0):
        J_r = -(1/mu_0) * dB_phi/dz
        J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
        J_z = (1/mu_0) * (1/r) * d(r*B_phi)/dr
            = (1/mu_0) * (B_phi/r + dB_phi/dr)

    At r=0, uses L'Hopital's rule:
        J_z[0,:] = (1/mu_0) * 2 * dB_phi/dr[0,:]
    """
    # J_r = -(1/mu_0) * dB_phi/dz
    dB_phi_dz = (jnp.roll(B_phi, -1, axis=1) - jnp.roll(B_phi, 1, axis=1)) / (2 * dz)
    J_r = -(1.0 / MU0) * dB_phi_dz

    # J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
    dB_r_dz = (jnp.roll(B_r, -1, axis=1) - jnp.roll(B_r, 1, axis=1)) / (2 * dz)
    dB_z_dr = (jnp.roll(B_z, -1, axis=0) - jnp.roll(B_z, 1, axis=0)) / (2 * dr)
    J_phi = (1.0 / MU0) * (dB_r_dz - dB_z_dr)

    # J_z = (1/mu_0) * (B_phi/r + dB_phi/dr)
    dB_phi_dr = (jnp.roll(B_phi, -1, axis=0) - jnp.roll(B_phi, 1, axis=0)) / (2 * dr)

    # Handle r=0 singularity
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    J_z = (1.0 / MU0) * jnp.where(
        r > 1e-10,
        B_phi / r_safe + dB_phi_dr,
        2.0 * dB_phi_dr  # L'Hopital at r=0
    )

    return J_r, J_phi, J_z
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_physics.py::TestCylindricalCurl::test_jz_includes_b_theta_over_r_term -v`

Expected: PASS

**Step 5: Run all Extended MHD tests**

Run: `py -m pytest tests/test_extended_mhd.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add jax_frc/models/extended_mhd.py tests/test_operators_physics.py
git commit -m "fix(extended_mhd): correct cylindrical curl J_z formula

Add missing B_phi/r term to J_z computation. The cylindrical curl
formula is J_z = (1/r)*d(r*B_phi)/dr = B_phi/r + dB_phi/dr.
Also handle r=0 singularity with L'Hopital's rule.

Fixes critical O(1) error near magnetic axis."
```

---

## Task 2: Fix Hybrid Kinetic E-field (Add Hall Term)

The `compute_rhs` method in `jax_frc/models/hybrid_kinetic.py` is missing the Hall term `(J × B)/(ne)`.

**Files:**
- Modify: `jax_frc/models/hybrid_kinetic.py:70-106`
- Modify: `jax_frc/models/hybrid_kinetic.py:59-68` (add geometry parameter)
- Test: `tests/test_operators_physics.py` (append)

**Step 1: Write failing test for Hall term**

Append to `tests/test_operators_physics.py`:

```python
class TestHybridHallTerm:
    """Tests for Hall term in Hybrid Kinetic E-field."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return Geometry(
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

    @pytest.fixture
    def model(self):
        """Create HybridKinetic model."""
        from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium
        equilibrium = RigidRotorEquilibrium(n0=1e19, T0=1000.0, Omega=1e5)
        return HybridKinetic(equilibrium=equilibrium, eta=1e-6)

    def test_e_field_includes_hall_term(self, geometry, model):
        """E-field should include (J × B)/(ne) Hall term.

        With uniform B_z and J_theta (from curl of B_z(r)):
            Hall_r = J_theta * B_z / (ne)
            Hall_z = -J_r * B_theta / (ne) = 0 (if B_theta = 0)

        The Hall term should contribute to E_r.
        """
        from jax_frc.core.state import State

        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Create state with non-trivial B field that produces current
        # B_z = B0 * exp(-r^2) generates J_theta from dB_z/dr
        B0 = 0.1  # Tesla
        B_r = jnp.zeros((nr, nz))
        B_phi = jnp.zeros((nr, nz))
        B_z = B0 * jnp.exp(-r**2) * jnp.ones((1, nz))
        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        n = jnp.ones((nr, nz)) * 1e19
        p = jnp.ones((nr, nz)) * 1e3  # Uniform pressure (no gradient contribution)
        v = jnp.zeros((nr, nz, 3))
        psi = jnp.zeros((nr, nz))
        E = jnp.zeros((nr, nz, 3))

        state = State(
            psi=psi, B=B, v=v, n=n, p=p, E=E,
            time=0.0, step=0, particles=None
        )

        # Compute RHS which includes E-field calculation
        rhs = model.compute_rhs(state, geometry)

        # The E field should have non-zero E_r from Hall term
        # J_theta ~ -dB_z/dr = 2*r*B0*exp(-r^2) (positive for r>0)
        # Hall_r = J_theta * B_z / (ne) should be non-zero
        E_r = rhs.E[:, :, 0]

        # Check that E_r is non-zero in interior (Hall contribution)
        interior_E_r = E_r[3:-3, 3:-3]
        max_E_r = float(jnp.max(jnp.abs(interior_E_r)))

        # With eta=1e-6 and B~0.1T, the resistive term is small
        # Hall term should dominate: |E_r| ~ |J*B|/(ne) ~ (B/mu0/L) * B / (n*e)
        # ~ (0.1 / 1.26e-6 / 0.1) * 0.1 / (1e19 * 1.6e-19) ~ 500 V/m
        assert max_E_r > 10.0, f"E_r too small ({max_E_r:.1f} V/m), Hall term likely missing"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_physics.py::TestHybridHallTerm::test_e_field_includes_hall_term -v`

Expected: FAIL with assertion that E_r is too small

**Step 3: Add Hall term to `compute_rhs`**

In `jax_frc/models/hybrid_kinetic.py`, replace the `compute_rhs` method:

```python
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    """Compute time derivatives for hybrid model.

    This computes the RHS for the electromagnetic fields using Faraday's law
    with E from the electron fluid equation (generalized Ohm's law):

        E = -v_e × B + ηJ - ∇p_e/(ne)
          = (J × B)/(ne) + ηJ - ∇p_e/(ne)

    where we used v_e = v_i - J/(ne) and quasi-neutrality.
    """
    dr, dz = geometry.dr, geometry.dz
    r = geometry.r_grid

    # Compute ion current from particles
    J_i = self.deposit_current(state, geometry)

    # For quasi-neutrality, J_total = J_i (electrons carry return current)
    J_total = J_i

    # Safe density to avoid division by zero
    n = jnp.maximum(state.n, 1e16)
    ne = n * QE

    # Extract field components
    B_r = state.B[:, :, 0]
    B_phi = state.B[:, :, 1]
    B_z = state.B[:, :, 2]
    J_r = J_total[:, :, 0]
    J_phi = J_total[:, :, 1]
    J_z = J_total[:, :, 2]

    # Hall term: (J × B) / (ne)
    hall_r = (J_phi * B_z - J_z * B_phi) / ne
    hall_phi = (J_z * B_r - J_r * B_z) / ne
    hall_z = (J_r * B_phi - J_phi * B_r) / ne

    # Electron pressure gradient term: -∇p_e/(ne)
    dp_dr = (jnp.roll(state.p, -1, axis=0) - jnp.roll(state.p, 1, axis=0)) / (2*dr)
    dp_dz = (jnp.roll(state.p, -1, axis=1) - jnp.roll(state.p, 1, axis=1)) / (2*dz)

    # Resistive term: η*J
    # Combined E-field: E = Hall + η*J - ∇p_e/(ne)
    E_r = hall_r + self.eta * J_r - dp_dr / ne
    E_phi = hall_phi + self.eta * J_phi
    E_z = hall_z + self.eta * J_z - dp_dz / ne

    E = jnp.stack([E_r, E_phi, E_z], axis=-1)

    # Faraday's law: dB/dt = -curl(E)
    dB_r, dB_phi, dB_z = self._compute_curl_E(E_r, E_phi, E_z, dr, dz)
    dB = jnp.stack([-dB_r, -dB_phi, -dB_z], axis=-1)

    # Store E in state for particle pushing
    return state.replace(B=dB, E=E)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_physics.py::TestHybridHallTerm::test_e_field_includes_hall_term -v`

Expected: PASS

**Step 5: Run all Hybrid Kinetic tests**

Run: `py -m pytest tests/test_hybrid_kinetic.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add jax_frc/models/hybrid_kinetic.py tests/test_operators_physics.py
git commit -m "fix(hybrid_kinetic): add missing Hall term to E-field

The generalized Ohm's law for hybrid model is:
  E = (J × B)/(ne) + ηJ - ∇p_e/(ne)

Previously only computed E = ηJ - ∇p_e/(ne), missing the Hall term
which is critical for FRC physics (tilt stabilization, wave propagation).

Fixes critical physics bug in Hybrid Kinetic model."
```

---

## Task 3: Add Krook Collision Operator to Hybrid Kinetic

Delta-f PIC requires collisions to prevent unbounded weight growth.

**Files:**
- Modify: `jax_frc/models/hybrid_kinetic.py:59-68` (add collision_frequency parameter)
- Modify: `jax_frc/models/hybrid_kinetic.py:161-203` (add collision in push_particles)
- Test: `tests/test_operators_physics.py` (append)

**Step 1: Write failing test for collision operator**

Append to `tests/test_operators_physics.py`:

```python
class TestHybridCollisions:
    """Tests for collision operator in Hybrid Kinetic."""

    def test_krook_collision_relaxes_weights(self):
        """Krook collision operator should relax weights toward zero.

        dw/dt = -ν * w  =>  w(t) = w(0) * exp(-ν*t)

        After time t = 1/ν, weights should decay by factor of ~1/e.
        """
        from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium
        from jax_frc.core.state import State, ParticleState
        from jax_frc.core.geometry import Geometry
        import jax.random as random

        # Setup
        geometry = Geometry(
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=8, nz=16
        )

        nu_collision = 1e4  # 1/s collision frequency
        equilibrium = RigidRotorEquilibrium(n0=1e19, T0=1000.0, Omega=1e5)
        model = HybridKinetic(
            equilibrium=equilibrium,
            eta=1e-6,
            collision_frequency=nu_collision
        )

        # Create particles with non-zero weights
        n_particles = 100
        key = random.PRNGKey(42)
        keys = random.split(key, 4)

        r = random.uniform(keys[0], (n_particles,), minval=0.1, maxval=0.9)
        theta = random.uniform(keys[1], (n_particles,), minval=0, maxval=2*jnp.pi)
        z = random.uniform(keys[2], (n_particles,), minval=-0.8, maxval=0.8)
        x = jnp.stack([r, theta, z], axis=-1)

        v_thermal = 1e5  # m/s
        v = random.normal(keys[3], (n_particles, 3)) * v_thermal

        # Start with weights = 0.5
        w_initial = jnp.ones(n_particles) * 0.5

        particles = ParticleState(x=x, v=v, w=w_initial, species="ion")

        # Create state with uniform fields (minimal E,B evolution)
        nr, nz = geometry.nr, geometry.nz
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.01)  # Small uniform B_z
        E = jnp.zeros((nr, nz, 3))

        state = State(
            psi=jnp.zeros((nr, nz)),
            B=B, v=jnp.zeros((nr, nz, 3)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            E=E, time=0.0, step=0,
            particles=particles
        )

        # Run for time = 1/nu (one collision time)
        dt = 1e-6  # 1 microsecond
        n_steps = int(1.0 / (nu_collision * dt))  # ~100 steps for 1 collision time

        for _ in range(n_steps):
            state = model.push_particles(state, geometry, dt)

        # Check weight decay
        w_final = state.particles.w
        mean_w_initial = float(jnp.mean(jnp.abs(w_initial)))
        mean_w_final = float(jnp.mean(jnp.abs(w_final)))

        # After 1 collision time, should decay to ~1/e ≈ 0.37 of initial
        decay_ratio = mean_w_final / mean_w_initial
        expected_ratio = jnp.exp(-1.0)  # ~0.37

        # Allow 20% tolerance for finite timestep effects
        assert decay_ratio < 0.5, f"Weights not decaying: ratio = {decay_ratio:.3f}"
        assert decay_ratio > 0.25, f"Weights decaying too fast: ratio = {decay_ratio:.3f}"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_physics.py::TestHybridCollisions::test_krook_collision_relaxes_weights -v`

Expected: FAIL (collision_frequency parameter not recognized, or weights don't decay)

**Step 3: Add collision_frequency parameter to HybridKinetic**

In `jax_frc/models/hybrid_kinetic.py`, modify the dataclass:

```python
@dataclass
class HybridKinetic(PhysicsModel):
    """Hybrid kinetic model: kinetic ions + fluid electrons.

    Uses delta-f PIC method for ions to reduce statistical noise.
    Electrons are treated as a massless neutralizing fluid.

    Attributes:
        equilibrium: Background distribution for delta-f
        eta: Resistivity (Ohm·m)
        collision_frequency: Ion-ion collision frequency (1/s) for Krook operator.
            Typical FRC values: 1e3-1e5 s^-1. Set to 0 to disable collisions.
    """

    equilibrium: RigidRotorEquilibrium
    eta: float = 1e-6
    collision_frequency: float = 0.0  # Krook collision frequency (1/s)
```

**Step 4: Add collision operator to push_particles**

In `jax_frc/models/hybrid_kinetic.py`, modify `push_particles`:

```python
def push_particles(self, state: State, geometry: Geometry, dt: float) -> State:
    """Advance particles using Boris pusher and update weights.

    Includes Krook collision operator: dw/dt = -ν*w
    This relaxes the delta-f weights toward zero, preventing secular growth.
    """
    if state.particles is None:
        return state

    particles = state.particles
    x = particles.x
    v = particles.v
    w = particles.w

    # Geometry parameters for interpolation
    geom_params = (geometry.r_min, geometry.r_max,
                   geometry.z_min, geometry.z_max,
                   geometry.nr, geometry.nz)

    # Interpolate E and B fields to particle positions
    E_p = interpolate_field_to_particles(state.E, x, geom_params)
    B_p = interpolate_field_to_particles(state.B, x, geom_params)

    # Boris push
    x_new, v_new = boris_push(x, v, E_p, B_p, QE, MI, dt)

    # Compute acceleration for weight evolution
    a = QE / MI * (E_p + jnp.cross(v, B_p))

    # Weight evolution: dw/dt = -(1-w) * d(ln f0)/dt
    r = x[:, 0]
    vr, vtheta, vz = v[:, 0], v[:, 1], v[:, 2]
    ar, atheta, az = a[:, 0], a[:, 1], a[:, 2]

    dlnf0_dt = self.equilibrium.d_ln_f0_dt(r, vr, vtheta, vz, ar, atheta, az)
    dw_physics = -(1 - w) * dlnf0_dt

    # Krook collision operator: dw/dt = -ν*w
    # Exact solution: w(t+dt) = w(t) * exp(-ν*dt)
    if self.collision_frequency > 0:
        collision_decay = jnp.exp(-self.collision_frequency * dt)
        w_new = (w + dw_physics * dt) * collision_decay
    else:
        w_new = w + dw_physics * dt

    # Clip weights to [-1, 1] for numerical stability
    w_new = jnp.clip(w_new, -1.0, 1.0)

    # Create new particle state
    new_particles = ParticleState(
        x=x_new,
        v=v_new,
        w=w_new,
        species=particles.species
    )

    return state.replace(particles=new_particles)
```

**Step 5: Update from_config to include collision_frequency**

In `jax_frc/models/hybrid_kinetic.py`, modify `from_config`:

```python
@classmethod
def from_config(cls, config: dict) -> "HybridKinetic":
    """Create from configuration dictionary."""
    eq_config = config.get("equilibrium", {})
    equilibrium = RigidRotorEquilibrium(
        n0=float(eq_config.get("n0", 1e19)),
        T0=float(eq_config.get("T0", 1000.0)),
        Omega=float(eq_config.get("Omega", 1e5))
    )

    eta = float(config.get("eta", 1e-6))
    collision_frequency = float(config.get("collision_frequency", 0.0))

    return cls(
        equilibrium=equilibrium,
        eta=eta,
        collision_frequency=collision_frequency
    )
```

**Step 6: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_physics.py::TestHybridCollisions::test_krook_collision_relaxes_weights -v`

Expected: PASS

**Step 7: Run all Hybrid Kinetic tests**

Run: `py -m pytest tests/test_hybrid_kinetic.py -v`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add jax_frc/models/hybrid_kinetic.py tests/test_operators_physics.py
git commit -m "feat(hybrid_kinetic): add Krook collision operator

Add collision_frequency parameter for Krook collision operator:
  dw/dt = -ν*w

This relaxes delta-f weights toward zero, preventing secular growth
that would otherwise make simulations unphysical after a few collision
times. Implemented with exact exponential decay for stability.

Default is 0 (no collisions) for backward compatibility.
Typical FRC values: 1e3-1e5 s^-1."
```

---

## Task 4: Add Divergence Cleaning for B-field

Numerical errors accumulate ∇·B ≠ 0. Add projection-based cleaning.

**Files:**
- Create: `jax_frc/solvers/divergence_cleaning.py`
- Modify: `jax_frc/solvers/semi_implicit.py:29-53`
- Test: `tests/test_operators_physics.py` (append)

**Step 1: Write failing test for divergence cleaning**

Append to `tests/test_operators_physics.py`:

```python
class TestDivergenceCleaning:
    """Tests for ∇·B divergence cleaning."""

    def test_projection_reduces_div_b(self):
        """Projection method should reduce |∇·B| to near zero.

        For B with non-zero divergence, cleaning via:
            B_clean = B - ∇φ, where ∇²φ = ∇·B
        should produce ∇·B_clean ≈ 0.
        """
        from jax_frc.solvers.divergence_cleaning import clean_divergence_b
        from jax_frc.core.geometry import Geometry

        geometry = Geometry(
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=32, nz=64
        )

        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Create B field with non-zero divergence
        # B_r = r, B_z = z gives div(B) = 2 + 1 = 3 in Cartesian
        # In cylindrical: div(B) = (1/r)*d(r*B_r)/dr + dB_z/dz
        #                        = (1/r)*(2r) + 1 = 3
        B_r = r * jnp.ones((1, nz))
        B_phi = jnp.zeros((nr, nz))
        B_z = geometry.z_grid * jnp.ones((nr, 1))
        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        # Compute initial divergence
        from jax_frc.operators import divergence_cylindrical
        div_B_initial = divergence_cylindrical(B_r, B_z, dr, dz, r)
        max_div_initial = float(jnp.max(jnp.abs(div_B_initial[2:-2, 2:-2])))

        # Clean divergence
        B_clean = clean_divergence_b(B, geometry)

        # Compute cleaned divergence
        B_r_clean = B_clean[:, :, 0]
        B_z_clean = B_clean[:, :, 2]
        div_B_clean = divergence_cylindrical(B_r_clean, B_z_clean, dr, dz, r)
        max_div_clean = float(jnp.max(jnp.abs(div_B_clean[2:-2, 2:-2])))

        # Divergence should be reduced by at least 10x
        reduction = max_div_initial / max(max_div_clean, 1e-20)
        assert reduction > 10, f"Insufficient cleaning: {max_div_initial:.2e} -> {max_div_clean:.2e}"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_physics.py::TestDivergenceCleaning::test_projection_reduces_div_b -v`

Expected: FAIL (module not found)

**Step 3: Create divergence cleaning module**

Create `jax_frc/solvers/divergence_cleaning.py`:

```python
"""Divergence cleaning methods for magnetic field evolution.

Numerical errors in field evolution accumulate ∇·B ≠ 0, which causes
unphysical parallel forces and energy conservation violations.

This module provides projection-based cleaning:
    B_clean = B - ∇φ, where ∇²φ = ∇·B
"""

import jax.numpy as jnp
from jax import jit
from jax.scipy.sparse.linalg import cg
from typing import Tuple

from jax_frc.core.geometry import Geometry
from jax_frc.operators import divergence_cylindrical, gradient_r, gradient_z


@jit
def clean_divergence_b(B: jnp.ndarray, geometry: Geometry,
                       tol: float = 1e-6, maxiter: int = 100) -> jnp.ndarray:
    """Clean divergence from magnetic field using projection method.

    Solves: ∇²φ = ∇·B in cylindrical coordinates
    Then:   B_clean = B - ∇φ

    The Laplacian in cylindrical coords (axisymmetric):
        ∇²φ = (1/r)∂(r∂φ/∂r)/∂r + ∂²φ/∂z²

    Args:
        B: Magnetic field array of shape (nr, nz, 3)
        geometry: Computational geometry
        tol: Conjugate gradient tolerance
        maxiter: Maximum CG iterations

    Returns:
        Cleaned magnetic field with same shape
    """
    nr, nz = geometry.nr, geometry.nz
    dr, dz = geometry.dr, geometry.dz
    r = geometry.r_grid

    B_r = B[:, :, 0]
    B_phi = B[:, :, 1]
    B_z = B[:, :, 2]

    # Compute divergence
    div_B = divergence_cylindrical(B_r, B_z, dr, dz, r)

    # Solve ∇²φ = div_B using conjugate gradient
    # Define the Laplacian operator as a function
    def laplacian_op(phi_flat):
        phi = phi_flat.reshape((nr, nz))
        lap = _cylindrical_laplacian(phi, dr, dz, r)
        # Apply Neumann BC (zero gradient at boundaries)
        lap = lap.at[0, :].set(0)
        lap = lap.at[-1, :].set(0)
        lap = lap.at[:, 0].set(0)
        lap = lap.at[:, -1].set(0)
        return lap.flatten()

    # Initial guess
    phi_init = jnp.zeros(nr * nz)

    # RHS with zero boundary
    rhs = div_B.copy()
    rhs = rhs.at[0, :].set(0)
    rhs = rhs.at[-1, :].set(0)
    rhs = rhs.at[:, 0].set(0)
    rhs = rhs.at[:, -1].set(0)

    # Solve with CG
    phi_flat, info = cg(laplacian_op, rhs.flatten(), x0=phi_init,
                        tol=tol, maxiter=maxiter)
    phi = phi_flat.reshape((nr, nz))

    # Compute gradient of phi
    dphi_dr = gradient_r(phi, dr)
    dphi_dz = gradient_z(phi, dz)

    # Subtract gradient from B
    B_r_clean = B_r - dphi_dr
    B_z_clean = B_z - dphi_dz
    # B_phi unchanged (no theta dependence in axisymmetric)

    return jnp.stack([B_r_clean, B_phi, B_z_clean], axis=-1)


def _cylindrical_laplacian(phi: jnp.ndarray, dr: float, dz: float,
                           r: jnp.ndarray) -> jnp.ndarray:
    """Compute cylindrical Laplacian: (1/r)∂(r∂φ/∂r)/∂r + ∂²φ/∂z².

    Expanded: ∂²φ/∂r² + (1/r)∂φ/∂r + ∂²φ/∂z²

    At r=0, use L'Hopital: lim(r->0) (1/r)∂φ/∂r = ∂²φ/∂r²
    So: ∇²φ[r=0] = 2∂²φ/∂r² + ∂²φ/∂z²
    """
    # Second derivatives
    phi_rr = (jnp.roll(phi, -1, axis=0) - 2*phi + jnp.roll(phi, 1, axis=0)) / dr**2
    phi_zz = (jnp.roll(phi, -1, axis=1) - 2*phi + jnp.roll(phi, 1, axis=1)) / dz**2

    # First derivative for (1/r)∂φ/∂r term
    phi_r = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2*dr)

    # Regular formula away from axis
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    lap = jnp.where(r > 1e-10, phi_rr + phi_r/r_safe + phi_zz, 0.0)

    # L'Hopital at r=0
    lap = lap.at[0, :].set(2*phi_rr[0, :] + phi_zz[0, :])

    return lap
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_physics.py::TestDivergenceCleaning::test_projection_reduces_div_b -v`

Expected: PASS

**Step 5: Add divergence cleaning to SemiImplicitSolver**

In `jax_frc/solvers/semi_implicit.py`, modify the `step` method:

```python
def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
    """Advance state using semi-implicit Hall damping with divergence cleaning."""
    from jax_frc.solvers.divergence_cleaning import clean_divergence_b

    # Get explicit RHS
    rhs = model.compute_rhs(state, geometry)

    # For psi-based models (Resistive MHD), use explicit stepping
    new_psi = state.psi + dt * rhs.psi

    # For B-based models (Extended MHD), apply semi-implicit correction
    if jnp.any(rhs.B != 0):
        new_B = self._semi_implicit_B_update(state, rhs, dt, geometry)
        # Clean divergence every step
        new_B = clean_divergence_b(new_B, geometry)
    else:
        new_B = state.B

    # E field (for Hybrid)
    new_E = rhs.E if jnp.any(rhs.E != 0) else state.E

    new_state = state.replace(
        psi=new_psi,
        B=new_B,
        E=new_E,
        time=state.time + dt,
        step=state.step + 1
    )
    return model.apply_constraints(new_state, geometry)
```

**Step 6: Update `__init__.py` exports**

In `jax_frc/solvers/__init__.py`, add:

```python
from jax_frc.solvers.divergence_cleaning import clean_divergence_b
```

**Step 7: Run all solver tests**

Run: `py -m pytest tests/ -k "solver or extended_mhd" -v`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add jax_frc/solvers/divergence_cleaning.py jax_frc/solvers/semi_implicit.py jax_frc/solvers/__init__.py tests/test_operators_physics.py
git commit -m "feat(solvers): add projection-based divergence cleaning

Implement ∇·B cleaning via projection method:
  B_clean = B - ∇φ, where ∇²φ = ∇·B

Uses conjugate gradient solver with cylindrical Laplacian that
handles r=0 singularity via L'Hopital's rule.

Applied automatically in SemiImplicitSolver after each B-field update
to prevent accumulation of numerical divergence errors."
```

---

## Task 5: Run Full Test Suite and Verify

**Files:**
- All modified files

**Step 1: Run complete test suite**

Run: `py -m pytest tests/ -v`

Expected: All tests PASS

**Step 2: Run examples to verify models still work**

Run: `python examples.py`

Expected: All examples complete without error

**Step 3: Create summary commit if all passes**

If any failures occurred, fix them before this commit.

```bash
git log --oneline -4
```

Verify the 4 commits from this plan are present.

---

## Summary

This plan addresses the top 4 priority correctness fixes:

| Task | Fix | Status |
|------|-----|--------|
| 1 | Cylindrical curl J_z formula | |
| 2 | Hall term in Hybrid E-field | |
| 3 | Krook collision operator | |
| 4 | Divergence cleaning | |
| 5 | Full verification | |

After completing this plan, the physics models will be correct for FRC simulations and ready for validation against published benchmarks.
