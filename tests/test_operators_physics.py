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
