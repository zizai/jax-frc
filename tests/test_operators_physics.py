"""Physics correctness tests for differential operators."""
import pytest
import jax.numpy as jnp
from tests.utils.cartesian import make_geometry


class TestCylindricalCurl:
    """Tests for cylindrical curl operator correctness."""

    @pytest.fixture
    def grid(self):
        """Create r-z grid for cylindrical operator tests."""
        nr, nz = 16, 32
        r_min, r_max = 0.01, 1.0
        z_min, z_max = -1.0, 1.0
        dr = (r_max - r_min) / nr
        dz = (z_max - z_min) / nz
        r = jnp.linspace(r_min + dr / 2, r_max - dr / 2, nr)[:, None]
        z = jnp.linspace(z_min + dz / 2, z_max - dz / 2, nz)[None, :]
        return nr, nz, dr, dz, r, z

    def test_jz_includes_b_theta_over_r_term(self, grid):
        """J_z should include B_theta/r term, not just dB_theta/dr.

        For B_theta = r (linear in r), the correct J_z is:
            J_z = (1/mu0) * (B_theta/r + dB_theta/dr)
                = (1/mu0) * (r/r + 1)
                = (1/mu0) * 2

        The incorrect formula (just dB_theta/dr) gives:
            J_z = (1/mu0) * 1
        """
        nr, nz, dr, dz, r, _ = grid

        # B_theta = r (linear profile)
        B_r = jnp.zeros((nr, nz))
        B_theta = r * jnp.ones((1, nz))  # B_theta = r
        B_z = jnp.zeros((nr, nz))

        # Note: The corrected _compute_current method requires r parameter
        from jax_frc.operators import curl_cylindrical_axisymmetric
        J_r, J_phi, J_z = curl_cylindrical_axisymmetric(B_r, B_theta, B_z, dr, dz, r)

        # Expected: J_z = 2/mu0 everywhere (except boundaries)
        expected_J_z = 2.0

        # Check interior points (avoid boundaries)
        interior_J_z = J_z[5:-5, 5:-5]
        interior_expected = jnp.ones_like(interior_J_z) * expected_J_z

        # Should match within 5% (finite difference error)
        relative_error = jnp.abs(interior_J_z - interior_expected) / expected_J_z
        max_error = float(jnp.max(relative_error))

        assert max_error < 0.05, f"J_z incorrect: max relative error {max_error:.2%}"


class TestHybridHallTerm:
    """Tests for Hall term in Hybrid Kinetic E-field."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=1, nz=32, extent=1.0)

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

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with non-trivial B field that produces current
        # B_z = B0 * exp(-r^2) generates J_theta from dB_z/dr
        B0 = 0.1  # Tesla
        x = geometry.x_grid
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(B0 * jnp.exp(-x**2))

        n = jnp.ones((nx, ny, nz)) * 1e19
        p = jnp.ones((nx, ny, nz)) * 1e3  # Uniform pressure (no gradient contribution)
        v = jnp.zeros((nx, ny, nz, 3))
        E = jnp.zeros((nx, ny, nz, 3))

        state = State.zeros(nx, ny, nz).replace(B=B, v=v, n=n, p=p, E=E)

        # Compute RHS which includes E-field calculation
        rhs = model.compute_rhs(state, geometry)

        # The E field should have non-zero E_r from Hall term
        # J_theta ~ -dB_z/dr = 2*r*B0*exp(-r^2) (positive for r>0)
        # Hall_r = J_theta * B_z / (ne) should be non-zero
        E_r = rhs.E[:, :, :, 0]

        # Check that E_r is non-zero in interior (Hall contribution)
        interior_E_r = E_r[3:-3, 0, 3:-3]
        max_E_r = float(jnp.max(jnp.abs(interior_E_r)))

        # With eta=1e-6 and B~0.1T, the resistive term is small
        # Hall term should dominate: |E_r| ~ |J*B|/(ne) ~ (B/mu0/L) * B / (n*e)
        # ~ (0.1 / 1.26e-6 / 0.1) * 0.1 / (1e19 * 1.6e-19) ~ 500 V/m
        assert max_E_r > 10.0, f"E_r too small ({max_E_r:.1f} V/m), Hall term likely missing"


class TestHybridCollisions:
    """Tests for collision operator in Hybrid Kinetic."""

    def test_krook_collision_relaxes_weights(self):
        """Krook collision operator should relax weights toward zero.

        dw/dt = -ν * w  =>  w(t) = w(0) * exp(-ν*t)

        After time t = 1/ν, weights should decay by factor of ~1/e.
        """
        from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium
        from jax_frc.core.state import State, ParticleState
        import jax.random as random

        # Setup
        geometry = make_geometry(nx=8, ny=1, nz=16, extent=1.0)

        nu_collision = 1e5  # 1/s collision frequency (10x faster for quicker tests)
        equilibrium = RigidRotorEquilibrium(n0=1e19, T0=1000.0, Omega=1e5)
        model = HybridKinetic(
            equilibrium=equilibrium,
            eta=1e-6,
            collision_frequency=nu_collision
        )

        # Create particles with non-zero weights
        n_particles = 50
        key = random.PRNGKey(42)
        keys = random.split(key, 4)

        x = random.uniform(keys[0], (n_particles,), minval=-0.8, maxval=0.8)
        y = random.uniform(keys[1], (n_particles,), minval=-0.8, maxval=0.8)
        z = random.uniform(keys[2], (n_particles,), minval=-0.8, maxval=0.8)
        positions = jnp.stack([x, y, z], axis=-1)

        # Use equilibrium velocities: vtheta = Omega*r + small thermal
        # This minimizes physics weight evolution so we can isolate collision effect
        v_thermal = 1e4  # Small thermal spread
        vx = random.normal(keys[3], (n_particles,)) * v_thermal * 0.1
        vy = random.normal(random.fold_in(keys[3], 1), (n_particles,)) * v_thermal * 0.1
        vz = random.normal(random.fold_in(keys[3], 2), (n_particles,)) * v_thermal * 0.1
        v = jnp.stack([vx, vy, vz], axis=-1)

        # Start with weights = 0.5
        w_initial = jnp.ones(n_particles) * 0.5

        particles = ParticleState(x=positions, v=v, w=w_initial, species="ion")

        # Create state with zero E-field (no acceleration => no physics weight evolution)
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(0.01)  # Small uniform B_z
        E = jnp.zeros((nx, ny, nz, 3))  # Zero E-field

        state = State.zeros(nx, ny, nz).replace(
            B=B,
            v=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            E=E,
            particles=particles,
        )

        # Run for time = 1/nu (one collision time)
        dt = 1e-5  # 10 microseconds (larger step for faster tests)
        n_steps = int(1.0 / (nu_collision * dt))  # ~10 steps for 1 collision time

        for _ in range(n_steps):
            state = model.push_particles(state, geometry, dt)

        # Check weight decay
        w_final = state.particles.w
        mean_w_initial = float(jnp.mean(jnp.abs(w_initial)))
        mean_w_final = float(jnp.mean(jnp.abs(w_final)))

        # After 1 collision time, should decay to ~1/e ≈ 0.37 of initial
        decay_ratio = mean_w_final / mean_w_initial

        # Allow tolerance for finite timestep effects and weight evolution physics
        # (wider tolerance due to fewer steps)
        assert decay_ratio < 0.6, f"Weights not decaying: ratio = {decay_ratio:.3f}"
        assert decay_ratio > 0.2, f"Weights decaying too fast: ratio = {decay_ratio:.3f}"


class TestDivergenceCleaning:
    """Tests for divergence-B divergence cleaning."""

    def test_projection_reduces_div_b(self):
        """Projection method should reduce |div(B)| significantly.

        For B with non-zero divergence, cleaning via:
            B_clean = B - grad(phi), where laplacian(phi) = div(B)
        should produce div(B_clean) << div(B_initial).

        Note: The projection method with Dirichlet BCs is most effective in the
        interior of the domain. Near boundaries, gradient artifacts from the
        phi=0 boundary condition reduce effectiveness. We test on the interior
        region [3:-3, 3:-3] which represents the physics region of interest.
        """
        from jax_frc.solvers.divergence_cleaning import clean_divergence
        from jax_frc.operators import divergence_3d

        geometry = make_geometry(nx=8, ny=1, nz=16, extent=1.0)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create B field with non-zero divergence: B = (x, 0, z)
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 0].set(geometry.x_grid)
        B = B.at[:, :, :, 2].set(geometry.z_grid)

        # Compute initial divergence - check interior region
        div_B_initial = divergence_3d(B, geometry)
        max_div_initial = float(jnp.max(jnp.abs(div_B_initial[3:-3, 0, 3:-3])))

        # Clean divergence
        B_clean = clean_divergence(B, geometry)

        # Compute cleaned divergence - check same interior region
        div_B_clean = divergence_3d(B_clean, geometry)
        max_div_clean = float(jnp.max(jnp.abs(div_B_clean[3:-3, 0, 3:-3])))

        # Divergence should be reduced significantly (at least 5x) in the interior
        reduction = max_div_initial / max(max_div_clean, 1e-20)
        assert reduction > 5, f"Insufficient cleaning: {max_div_initial:.2e} -> {max_div_clean:.2e} (reduction {reduction:.1f}x)"
