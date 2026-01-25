"""Tests for boundary conditions."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC


class TestTimeDependentMirrorBC:
    """Tests for time-dependent mirror field boundary condition."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0
        )

    @pytest.fixture
    def initial_state(self):
        return State.zeros(nr=20, nz=40)

    def test_no_change_at_t0(self, initial_state, geometry):
        """At t=0, boundary is unchanged (mirror_ratio = 1.0)."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=1.5,
            ramp_time=10.0,
            profile="cosine"
        )

        # State already has time=0.0 by default
        result = bc.apply(initial_state, geometry)

        # At t=0, psi at boundaries should be unchanged
        assert jnp.allclose(result.psi[:, 0], initial_state.psi[:, 0])
        assert jnp.allclose(result.psi[:, -1], initial_state.psi[:, -1])

    def test_full_compression_at_ramp_end(self, initial_state, geometry):
        """At t=ramp_time, mirror_ratio reaches final value."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=1.5,
            ramp_time=10.0,
            profile="cosine"
        )

        # Set state time to ramp_time
        state_at_end = initial_state.replace(time=10.0)
        result = bc.apply(state_at_end, geometry)

        # At t=ramp_time, boundary field should be at final value
        assert result is not None
        # Verify mirror ratio is at final value
        assert bc._compute_mirror_ratio(10.0) == 1.5

    def test_cosine_profile_smooth(self, initial_state, geometry):
        """Cosine profile gives smooth ramp."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=2.0,
            ramp_time=10.0,
            profile="cosine"
        )

        ratios = []
        for t in [0, 2.5, 5.0, 7.5, 10.0]:
            ratio = bc._compute_mirror_ratio(t)
            ratios.append(ratio)

        # Should be monotonically increasing
        assert ratios == sorted(ratios)
        # Should start at 1.0 and end at 2.0
        assert abs(ratios[0] - 1.0) < 0.01
        assert abs(ratios[-1] - 2.0) < 0.01
