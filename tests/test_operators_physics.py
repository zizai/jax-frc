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
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
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

        # Note: The corrected _compute_current method requires r parameter
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


class TestHybridHallTerm:
    """Tests for Hall term in Hybrid Kinetic E-field."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return Geometry(
            coord_system="cylindrical",
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

        # Create state with non-trivial B field that produces current
        # B_z = B0 * exp(-r^2) generates J_theta from dB_z/dr
        B0 = 0.1  # Tesla
        r = geometry.r_grid
        B_r = jnp.zeros((nr, nz))
        B_phi = jnp.zeros((nr, nz))
        B_z = B0 * jnp.exp(-r**2) * jnp.ones((1, nz))
        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        n = jnp.ones((nr, nz)) * 1e19
        p = jnp.ones((nr, nz)) * 1e3  # Uniform pressure (no gradient contribution)
        T = jnp.ones((nr, nz)) * 100.0  # Temperature in eV
        v = jnp.zeros((nr, nz, 3))
        psi = jnp.zeros((nr, nz))
        E = jnp.zeros((nr, nz, 3))

        state = State(
            psi=psi, B=B, v=v, n=n, p=p, T=T, E=E,
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
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=8, nz=16
        )

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

        r = random.uniform(keys[0], (n_particles,), minval=0.1, maxval=0.9)
        theta = random.uniform(keys[1], (n_particles,), minval=0, maxval=2*jnp.pi)
        z = random.uniform(keys[2], (n_particles,), minval=-0.8, maxval=0.8)
        x = jnp.stack([r, theta, z], axis=-1)

        # Use equilibrium velocities: vtheta = Omega*r + small thermal
        # This minimizes physics weight evolution so we can isolate collision effect
        v_thermal = 1e4  # Small thermal spread
        vr = random.normal(keys[3], (n_particles,)) * v_thermal * 0.1
        vz = random.normal(random.fold_in(keys[3], 1), (n_particles,)) * v_thermal * 0.1
        vtheta = equilibrium.Omega * r + random.normal(random.fold_in(keys[3], 2), (n_particles,)) * v_thermal * 0.1
        v = jnp.stack([vr, vtheta, vz], axis=-1)

        # Start with weights = 0.5
        w_initial = jnp.ones(n_particles) * 0.5

        particles = ParticleState(x=x, v=v, w=w_initial, species="ion")

        # Create state with zero E-field (no acceleration => no physics weight evolution)
        nr, nz = geometry.nr, geometry.nz
        B = jnp.zeros((nr, nz, 3))
        B = B.at[:, :, 2].set(0.01)  # Small uniform B_z
        E = jnp.zeros((nr, nz, 3))  # Zero E-field

        state = State(
            psi=jnp.zeros((nr, nz)),
            B=B, v=jnp.zeros((nr, nz, 3)),
            n=jnp.ones((nr, nz)) * 1e19,
            p=jnp.ones((nr, nz)) * 1e3,
            T=jnp.ones((nr, nz)) * 100.0,  # Temperature in eV
            E=E, time=0.0, step=0,
            particles=particles
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
        region [5:-5, 5:-5] which represents the physics region of interest.
        """
        from jax_frc.solvers.divergence_cleaning import clean_divergence_b
        from jax_frc.core.geometry import Geometry

        geometry = Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=1.0,
            z_min=-1.0, z_max=1.0,
            nr=16, nz=32
        )

        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Create B field with non-zero divergence
        # B_r = r, B_z = z gives div(B) != 0
        # In cylindrical: div(B) = (1/r)*d(r*B_r)/dr + dB_z/dz
        #                        = (1/r)*(2r) + 1 = 3
        B_r = r * jnp.ones((1, nz))
        B_phi = jnp.zeros((nr, nz))
        B_z = geometry.z_grid * jnp.ones((nr, 1))
        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        # Compute initial divergence - check interior region
        from jax_frc.operators import divergence_cylindrical
        div_B_initial = divergence_cylindrical(B_r, B_z, dr, dz, r)
        max_div_initial = float(jnp.max(jnp.abs(div_B_initial[5:-5, 5:-5])))

        # Clean divergence
        B_clean = clean_divergence_b(B, geometry)

        # Compute cleaned divergence - check same interior region
        B_r_clean = B_clean[:, :, 0]
        B_z_clean = B_clean[:, :, 2]
        div_B_clean = divergence_cylindrical(B_r_clean, B_z_clean, dr, dz, r)
        max_div_clean = float(jnp.max(jnp.abs(div_B_clean[5:-5, 5:-5])))

        # Divergence should be reduced significantly (at least 5x) in the interior
        reduction = max_div_initial / max(max_div_clean, 1e-20)
        assert reduction > 5, f"Insufficient cleaning: {max_div_initial:.2e} -> {max_div_clean:.2e} (reduction {reduction:.1f}x)"
