"""Tests for external field integration with physics models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestExternalFieldIntegration:
    """Tests for external field integration."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            nx=20,
            ny=4,
            nz=40,
            x_min=0.01,
            x_max=0.5,
            y_min=0.0,
            y_max=2 * jnp.pi,
            z_min=-1.0,
            z_max=1.0,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    @pytest.fixture
    def solenoid(self):
        return Solenoid(length=2.0, radius=0.6, n_turns=100, current=1000.0)

    def test_model_accepts_external_field(self, geometry, solenoid):
        """Model can be constructed with external_field parameter."""
        model = ResistiveMHD(eta=1e-6, external_field=solenoid)
        assert model.external_field is not None

    def test_external_field_adds_to_total(self, geometry, solenoid):
        """External B adds to equilibrium B in total field."""
        model = ResistiveMHD(eta=1e-6, external_field=solenoid)
        state = State.zeros(nx=20, ny=4, nz=40)

        # Get total B field
        B_x, B_z = model.get_total_B(state, geometry, t=0.0)

        # Should have contribution from external field
        # At center, solenoid should contribute ~B_z
        center_x, center_z = 10, 20
        center_y = geometry.ny // 2
        assert jnp.abs(B_z[center_x, center_y, center_z]) > 1e-6  # B_z component

    def test_no_external_field_default(self, geometry):
        """Without external_field, model uses only equilibrium B."""
        model = ResistiveMHD(eta=1e-6)
        assert model.external_field is None

    def test_total_B_without_external_field(self, geometry):
        """get_total_B works when external_field is None."""
        model = ResistiveMHD(eta=1e-6)
        state = State.zeros(nx=20, ny=4, nz=40)

        # Should not raise an error
        B_x, B_z = model.get_total_B(state, geometry, t=0.0)

        # With zero psi, B should be zero (or very small due to numerics)
        assert B_x.shape == (20, 4, 40)
        assert B_z.shape == (20, 4, 40)

    def test_total_B_with_nonzero_plasma_B(self, geometry, solenoid):
        """get_total_B includes contribution from plasma B."""
        model = ResistiveMHD(eta=1e-6, external_field=solenoid)

        # Create state with non-zero Bz (simple Gaussian profile)
        x = geometry.x_grid
        z = geometry.z_grid
        x0, z0 = 0.25, 0.0
        Bz = jnp.exp(-((x - x0)**2 + (z - z0)**2) / 0.1**2)
        B_plasma = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B_plasma = B_plasma.at[:, :, :, 2].set(Bz)

        state = State.zeros(nx=20, ny=4, nz=40).replace(B=B_plasma)

        # Get total B field
        B_x, B_z = model.get_total_B(state, geometry, t=0.0)

        # B should be non-zero from both psi and external field
        assert jnp.max(jnp.abs(B_x)) > 0
        assert jnp.max(jnp.abs(B_z)) > 0

    def test_external_field_contributes_to_total(self, geometry, solenoid):
        """Verify external field actually adds to the total."""
        model_with = ResistiveMHD(eta=1e-6, external_field=solenoid)
        model_without = ResistiveMHD(eta=1e-6, external_field=None)

        state = State.zeros(nx=20, ny=4, nz=40)

        B_x_with, B_z_with = model_with.get_total_B(state, geometry, t=0.0)
        B_x_without, B_z_without = model_without.get_total_B(state, geometry, t=0.0)

        # With external field should be different from without
        # (since psi=0, the difference should equal the external field)
        diff_z = jnp.max(jnp.abs(B_z_with - B_z_without))
        assert diff_z > 1e-6  # External field should make a difference

    def test_time_dependent_external_field(self, geometry):
        """get_total_B passes time to external field."""
        # Create solenoid with time-dependent current
        def current_func(t):
            return 1000.0 * (1.0 + t)

        solenoid = Solenoid(
            length=2.0, radius=0.6, n_turns=100,
            current=current_func
        )
        model = ResistiveMHD(eta=1e-6, external_field=solenoid)
        state = State.zeros(nx=20, ny=4, nz=40)

        # Get B at t=0 and t=1
        B_x_t0, B_z_t0 = model.get_total_B(state, geometry, t=0.0)
        B_x_t1, B_z_t1 = model.get_total_B(state, geometry, t=1.0)

        # B at t=1 should be ~2x B at t=0 (since current doubles)
        ratio = jnp.max(jnp.abs(B_z_t1)) / jnp.max(jnp.abs(B_z_t0))
        assert jnp.abs(ratio - 2.0) < 0.1  # Allow some tolerance


class TestExtendedMHDExternalField:
    """Tests for external field integration with ExtendedMHD model."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            nx=20,
            ny=4,
            nz=40,
            x_min=0.01,
            x_max=0.5,
            y_min=0.0,
            y_max=2 * jnp.pi,
            z_min=-1.0,
            z_max=1.0,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    @pytest.fixture
    def solenoid(self):
        return Solenoid(length=2.0, radius=0.6, n_turns=100, current=1000.0)

    @pytest.fixture
    def extended_mhd(self):
        from jax_frc.models.extended_mhd import ExtendedMHD
        return ExtendedMHD(eta=1e-6)

    def test_model_accepts_external_field(self, geometry, solenoid, extended_mhd):
        """ExtendedMHD can be constructed with external_field parameter."""
        from jax_frc.models.extended_mhd import ExtendedMHD
        model = ExtendedMHD(
            eta=1e-6,
            external_field=solenoid
        )
        assert model.external_field is not None

    def test_external_field_adds_to_total(self, geometry, solenoid):
        """External B adds to plasma B in total field."""
        from jax_frc.models.extended_mhd import ExtendedMHD
        model = ExtendedMHD(
            eta=1e-6,
            external_field=solenoid
        )
        state = State.zeros(nx=20, ny=4, nz=40)

        # Get total B field
        B_x, B_z = model.get_total_B(state, geometry, t=0.0)

        # Should have contribution from external field
        center_x, center_z = 10, 20
        center_y = geometry.ny // 2
        assert jnp.abs(B_z[center_x, center_y, center_z]) > 1e-6

    def test_no_external_field_default(self, geometry, extended_mhd):
        """Without external_field, model uses only plasma B."""
        assert extended_mhd.external_field is None

    def test_total_B_without_external_field(self, geometry, extended_mhd):
        """get_total_B works when external_field is None."""
        state = State.zeros(nx=20, ny=4, nz=40)

        # Should not raise an error
        B_x, B_z = extended_mhd.get_total_B(state, geometry, t=0.0)

        # With zero B in state, B should be zero
        assert B_x.shape == (20, 4, 40)
        assert B_z.shape == (20, 4, 40)

    def test_total_B_with_nonzero_plasma_B(self, geometry, solenoid):
        """get_total_B includes contribution from plasma B field."""
        from jax_frc.models.extended_mhd import ExtendedMHD
        model = ExtendedMHD(
            eta=1e-6,
            external_field=solenoid
        )

        # Create state with non-zero B field
        state = State.zeros(nx=20, ny=4, nz=40)
        B_plasma = jnp.ones((20, 4, 40, 3)) * 0.1  # 0.1 T uniform
        state = state.replace(B=B_plasma)

        # Get total B field
        B_x, B_z = model.get_total_B(state, geometry, t=0.0)

        # B should be non-zero from both plasma and external field
        assert jnp.max(jnp.abs(B_x)) > 0
        assert jnp.max(jnp.abs(B_z)) > 0

    def test_external_field_contributes_to_total(self, geometry, solenoid):
        """Verify external field actually adds to the total."""
        from jax_frc.models.extended_mhd import ExtendedMHD
        model_with = ExtendedMHD(
            eta=1e-6,
            external_field=solenoid
        )
        model_without = ExtendedMHD(
            eta=1e-6,
            external_field=None
        )

        state = State.zeros(nx=20, ny=4, nz=40)

        B_x_with, B_z_with = model_with.get_total_B(state, geometry, t=0.0)
        B_x_without, B_z_without = model_without.get_total_B(state, geometry, t=0.0)

        # With external field should be different from without
        diff_z = jnp.max(jnp.abs(B_z_with - B_z_without))
        assert diff_z > 1e-6

    def test_time_dependent_external_field(self, geometry):
        """get_total_B passes time to external field."""
        from jax_frc.models.extended_mhd import ExtendedMHD

        # Create solenoid with time-dependent current
        def current_func(t):
            return 1000.0 * (1.0 + t)

        solenoid = Solenoid(
            length=2.0, radius=0.6, n_turns=100,
            current=current_func
        )
        model = ExtendedMHD(
            eta=1e-6,
            external_field=solenoid
        )
        state = State.zeros(nx=20, ny=4, nz=40)

        # Get B at t=0 and t=1
        B_x_t0, B_z_t0 = model.get_total_B(state, geometry, t=0.0)
        B_x_t1, B_z_t1 = model.get_total_B(state, geometry, t=1.0)

        # B at t=1 should be ~2x B at t=0 (since current doubles)
        ratio = jnp.max(jnp.abs(B_z_t1)) / jnp.max(jnp.abs(B_z_t0))
        assert jnp.abs(ratio - 2.0) < 0.1
