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
