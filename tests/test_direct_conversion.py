"""Tests for direct induction energy conversion."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16, nz=32,
        r_min=0.1, r_max=0.5,
        z_min=-1.0, z_max=1.0,
    )


class TestConversionState:
    """Tests for ConversionState dataclass."""

    def test_conversion_state_creation(self):
        """Can create ConversionState."""
        from jax_frc.burn.conversion import ConversionState
        state = ConversionState(
            P_electric=1e6,
            V_induced=1000.0,
            dPsi_dt=0.1,
        )
        assert state.P_electric == 1e6


class TestDirectConversion:
    """Tests for direct induction conversion."""

    def test_direct_conversion_creation(self):
        """Can create DirectConversion."""
        from jax_frc.burn.conversion import DirectConversion
        dc = DirectConversion(
            coil_turns=100,
            coil_radius=0.6,
            circuit_resistance=0.1,
            coupling_efficiency=0.9,
        )
        assert dc.coil_turns == 100

    def test_induced_voltage_expanding_plasma(self, geometry):
        """Expanding plasma (decreasing B) induces positive voltage."""
        from jax_frc.burn.conversion import DirectConversion
        dc = DirectConversion(
            coil_turns=100,
            coil_radius=0.6,
            circuit_resistance=0.1,
            coupling_efficiency=1.0,
        )

        # B decreasing (plasma expanding against field)
        B_old = jnp.ones((geometry.nr, geometry.nz, 3)) * 1.0
        B_old = B_old.at[:, :, 2].set(1.0)  # Bz = 1 T

        B_new = jnp.ones((geometry.nr, geometry.nz, 3)) * 1.0
        B_new = B_new.at[:, :, 2].set(0.9)  # Bz decreased

        dt = 1e-6
        state = dc.compute_power(B_old, B_new, dt, geometry)

        # dPsi/dt < 0 (flux decreasing), V = -dPsi/dt > 0
        assert state.dPsi_dt < 0
        assert state.V_induced > 0
        assert state.P_electric > 0

    def test_zero_power_static_field(self, geometry):
        """No power extracted from static field."""
        from jax_frc.burn.conversion import DirectConversion
        dc = DirectConversion(
            coil_turns=100,
            coil_radius=0.6,
            circuit_resistance=0.1,
            coupling_efficiency=1.0,
        )

        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        dt = 1e-6
        state = dc.compute_power(B, B, dt, geometry)

        assert jnp.isclose(state.P_electric, 0, atol=1e-6)

    def test_power_scales_with_turns_squared(self, geometry):
        """Power ~ N^2 (voltage ~ N, power ~ V^2)."""
        from jax_frc.burn.conversion import DirectConversion

        B_old = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_old = B_old.at[:, :, 2].set(1.0)
        B_new = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_new = B_new.at[:, :, 2].set(0.9)
        dt = 1e-6

        dc1 = DirectConversion(coil_turns=100, coil_radius=0.6,
                               circuit_resistance=0.1, coupling_efficiency=1.0)
        dc2 = DirectConversion(coil_turns=200, coil_radius=0.6,
                               circuit_resistance=0.1, coupling_efficiency=1.0)

        P1 = dc1.compute_power(B_old, B_new, dt, geometry).P_electric
        P2 = dc2.compute_power(B_old, B_new, dt, geometry).P_electric

        # P2/P1 should be ~4 (200^2/100^2)
        assert jnp.isclose(P2 / P1, 4.0, rtol=0.01)
