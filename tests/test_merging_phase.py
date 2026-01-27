"""Tests for MergingPhase."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.configurations.phases.merging import MergingPhase
from jax_frc.configurations import timeout


class TestMergingPhase:
    """Tests for MergingPhase."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            nx=20,
            ny=4,
            nz=80,
            x_min=0.1,
            x_max=1.0,
            y_min=0.0,
            y_max=2 * jnp.pi,
            z_min=-4.0,
            z_max=4.0,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    @pytest.fixture
    def single_frc_state(self, geometry):
        """Create a single FRC equilibrium state."""
        state = State.zeros(nx=20, ny=4, nz=80)

        x = geometry.x_grid
        z = geometry.z_grid

        # Single FRC centered at z=0
        core = jnp.exp(-((x - 0.5)**2 + z**2) / 0.2)
        p = core * 0.5
        n = jnp.ones_like(core)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(core)
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))

        return state.replace(p=p, n=n, B=B, v=v)

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

        # Should have two pressure peaks
        p = result.p
        z_mid = p.shape[2] // 2

        left_max = jnp.max(p[:, :, :z_mid])
        right_max = jnp.max(p[:, :, z_mid:])

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
        z_mid = result.v.shape[2] // 2

        vz_left = result.v[:, :, z_mid // 2, 2]  # z component
        vz_right = result.v[:, :, z_mid + z_mid // 2, 2]

        # Should have opposite signs
        assert jnp.mean(vz_left) > 0
        assert jnp.mean(vz_right) < 0

    def test_step_hook_applies_compression_bc(self, single_frc_state, geometry):
        """MergingPhase.step_hook applies compression BC when configured."""
        phase = MergingPhase(
            name="merge",
            transition=timeout(10.0),
            separation=2.0,
            initial_velocity=0.1,
            compression={
                "base_field": 1.0,
                "mirror_ratio": 1.5,
                "ramp_time": 10.0,
                "profile": "cosine",
            },
        )

        # Setup creates the compression BC
        state = phase.setup(single_frc_state, geometry, {})
        assert phase._compression_bc is not None

        # step_hook should apply BC (no error means success)
        # Set time > 0 so compression has effect
        state_with_time = state.replace(time=5.0)
        result = phase.step_hook(state_with_time, geometry, t=5.0)
        assert result is not None
