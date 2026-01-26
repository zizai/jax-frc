"""Tests for pickup coil array."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


class TestPickupCoilArray:
    """Tests for PickupCoilArray."""

    def test_creation(self):
        """Can create PickupCoilArray."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf, jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([-0.5, 0.0, 0.5]),
            radii=jnp.array([0.4, 0.4, 0.4]),
            n_turns=jnp.array([100, 100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0, 1.0]),
        )
        assert pickup.n_coils == 3

    def test_flux_uniform_field(self, geometry):
        """Flux through coil in uniform Bz field."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.1]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )

        # Uniform Bz = 1 T field
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)  # Bz = 1 T

        Psi = pickup.compute_flux_linkages(B, geometry)

        # Expected: Psi = N * Bz * pi * r^2 = 100 * 1.0 * pi * 0.4^2
        # But we integrate over cells with r < 0.4, so it's approximate
        expected = 100 * 1.0 * jnp.pi * 0.4**2
        assert Psi.shape == (1,)
        # Allow some error due to discrete grid
        assert jnp.isclose(Psi[0], expected, rtol=0.1)

    def test_flux_multiple_coils(self, geometry):
        """Different coils see different flux based on position."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        # Two coils at different z positions
        pickup = PickupCoilArray(
            z_positions=jnp.array([-0.5, 0.5]),
            radii=jnp.array([0.4, 0.4]),
            n_turns=jnp.array([100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0]),
        )

        # Bz varies with z: stronger at z > 0
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        z_grid = geometry.z_grid
        Bz = jnp.where(z_grid > 0, 2.0, 1.0)
        B = B.at[:, :, 2].set(Bz)

        Psi = pickup.compute_flux_linkages(B, geometry)

        # Coil at z=0.5 should see ~2x the flux of coil at z=-0.5
        assert Psi[1] > Psi[0]
        assert jnp.isclose(Psi[1] / Psi[0], 2.0, rtol=0.2)

    def test_power_calculation(self):
        """Power extraction from currents."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0, 0.5]),
            radii=jnp.array([0.4, 0.4]),
            n_turns=jnp.array([100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0]),
        )

        I = jnp.array([10.0, 20.0])  # Currents in each coil

        P_load, P_dissipated = pickup.compute_power(I)

        # P_load = I^2 * R_load = 100*1 + 400*1 = 500 W
        assert jnp.isclose(jnp.sum(P_load), 500.0)
        # P_dissipated = I^2 * R = 100*0.1 + 400*0.1 = 50 W
        assert jnp.isclose(jnp.sum(P_dissipated), 50.0)
