"""Tests for external field integration with physics models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestExternalFieldIntegration:
    """Tests for external field integration."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.01, r_max=0.5,
            z_min=-1.0, z_max=1.0
        )

    @pytest.fixture
    def solenoid(self):
        return Solenoid(length=2.0, radius=0.6, n_turns=100, current=1000.0)

    @pytest.fixture
    def resistivity(self):
        return SpitzerResistivity(eta_0=1e-6)

    def test_model_accepts_external_field(self, geometry, solenoid, resistivity):
        """Model can be constructed with external_field parameter."""
        model = ResistiveMHD(resistivity=resistivity, external_field=solenoid)
        assert model.external_field is not None

    def test_external_field_adds_to_total(self, geometry, solenoid, resistivity):
        """External B adds to equilibrium B in total field."""
        model = ResistiveMHD(resistivity=resistivity, external_field=solenoid)
        state = State.zeros(nr=20, nz=40)

        # Get total B field
        B_r, B_z = model.get_total_B(state, geometry, t=0.0)

        # Should have contribution from external field
        # At center, solenoid should contribute ~B_z
        center_r, center_z = 10, 20
        assert jnp.abs(B_z[center_r, center_z]) > 1e-6  # B_z component

    def test_no_external_field_default(self, geometry, resistivity):
        """Without external_field, model uses only equilibrium B."""
        model = ResistiveMHD(resistivity=resistivity)
        assert model.external_field is None

    def test_total_B_without_external_field(self, geometry, resistivity):
        """get_total_B works when external_field is None."""
        model = ResistiveMHD(resistivity=resistivity)
        state = State.zeros(nr=20, nz=40)

        # Should not raise an error
        B_r, B_z = model.get_total_B(state, geometry, t=0.0)

        # With zero psi, B should be zero (or very small due to numerics)
        assert B_r.shape == (20, 40)
        assert B_z.shape == (20, 40)

    def test_total_B_with_nonzero_psi(self, geometry, solenoid, resistivity):
        """get_total_B includes contribution from psi."""
        model = ResistiveMHD(resistivity=resistivity, external_field=solenoid)

        # Create state with non-zero psi (simple Gaussian profile)
        r = geometry.r_grid
        z = geometry.z_grid
        r0, z0 = 0.25, 0.0
        psi = jnp.exp(-((r - r0)**2 + (z - z0)**2) / 0.1**2)

        state = State.zeros(nr=20, nz=40).replace(psi=psi)

        # Get total B field
        B_r, B_z = model.get_total_B(state, geometry, t=0.0)

        # B should be non-zero from both psi and external field
        assert jnp.max(jnp.abs(B_r)) > 0
        assert jnp.max(jnp.abs(B_z)) > 0

    def test_external_field_contributes_to_total(self, geometry, solenoid, resistivity):
        """Verify external field actually adds to the total."""
        model_with = ResistiveMHD(resistivity=resistivity, external_field=solenoid)
        model_without = ResistiveMHD(resistivity=resistivity, external_field=None)

        state = State.zeros(nr=20, nz=40)

        B_r_with, B_z_with = model_with.get_total_B(state, geometry, t=0.0)
        B_r_without, B_z_without = model_without.get_total_B(state, geometry, t=0.0)

        # With external field should be different from without
        # (since psi=0, the difference should equal the external field)
        diff_z = jnp.max(jnp.abs(B_z_with - B_z_without))
        assert diff_z > 1e-6  # External field should make a difference

    def test_time_dependent_external_field(self, geometry, resistivity):
        """get_total_B passes time to external field."""
        # Create solenoid with time-dependent current
        def current_func(t):
            return 1000.0 * (1.0 + t)

        solenoid = Solenoid(
            length=2.0, radius=0.6, n_turns=100,
            current=current_func
        )
        model = ResistiveMHD(resistivity=resistivity, external_field=solenoid)
        state = State.zeros(nr=20, nz=40)

        # Get B at t=0 and t=1
        B_r_t0, B_z_t0 = model.get_total_B(state, geometry, t=0.0)
        B_r_t1, B_z_t1 = model.get_total_B(state, geometry, t=1.0)

        # B at t=1 should be ~2x B at t=0 (since current doubles)
        ratio = jnp.max(jnp.abs(B_z_t1)) / jnp.max(jnp.abs(B_z_t0))
        assert jnp.abs(ratio - 2.0) < 0.1  # Allow some tolerance
