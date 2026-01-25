"""Tests for MergingPhase."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import timeout


class TestMergingPhase:
    """Tests for MergingPhase."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=80,
            r_min=0.1, r_max=1.0,
            z_min=-4.0, z_max=4.0
        )

    @pytest.fixture
    def single_frc_state(self, geometry):
        """Create a single FRC equilibrium state."""
        state = State.zeros(nr=20, nz=80)

        r = geometry.r_grid
        z = geometry.z_grid

        # Single FRC centered at z=0
        psi = jnp.exp(-((r - 0.5)**2 + z**2) / 0.2)
        p = psi * 0.5
        n = jnp.ones_like(psi)

        return state.replace(psi=psi, p=p, n=n)

    def test_setup_creates_two_frc_state(self, single_frc_state, geometry):
        """MergingPhase.setup creates two-FRC configuration."""
        phase = MergingPhase(
            name="merge",
            transition=timeout(10.0),
            separation=2.0,
            initial_velocity=0.1,
        )

        config = {}
        result = phase.setup(single_frc_state, geometry, config)

        # Should have two psi peaks
        psi = result.psi
        z_mid = psi.shape[1] // 2

        left_max = jnp.max(psi[:, :z_mid])
        right_max = jnp.max(psi[:, z_mid:])

        # Both halves should have significant flux
        assert left_max > 0.1
        assert right_max > 0.1

    def test_setup_applies_initial_velocity(self, single_frc_state, geometry):
        """MergingPhase.setup applies antisymmetric velocity."""
        phase = MergingPhase(
            name="merge",
            transition=timeout(10.0),
            separation=2.0,
            initial_velocity=0.2,
        )

        config = {}
        result = phase.setup(single_frc_state, geometry, config)

        # Left FRC should have +Vz, right FRC should have -Vz
        z_mid = result.v.shape[1] // 2

        vz_left = result.v[:, z_mid // 2, 2]  # z component
        vz_right = result.v[:, z_mid + z_mid // 2, 2]

        # Should have opposite signs
        assert jnp.mean(vz_left) > 0
        assert jnp.mean(vz_right) < 0
