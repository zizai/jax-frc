"""Tests for FluxCoupling module."""

import jax.numpy as jnp
import pytest
from jax import jit

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    """Standard geometry for tests."""
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


@pytest.fixture
def pickup_array():
    """Three-coil pickup array for tests."""
    from jax_frc.circuits import CircuitParams
    from jax_frc.circuits.pickup import PickupCoilArray

    params = CircuitParams(
        L=jnp.array([1e-3, 1e-3, 1e-3]),
        R=jnp.array([0.1, 0.1, 0.1]),
        C=jnp.array([jnp.inf, jnp.inf, jnp.inf]),
    )
    return PickupCoilArray(
        z_positions=jnp.array([-0.5, 0.0, 0.5]),
        radii=jnp.array([0.4, 0.4, 0.4]),
        n_turns=jnp.array([100, 100, 100]),
        params=params,
        load_resistance=jnp.array([1.0, 1.0, 1.0]),
    )


@pytest.fixture
def external_circuits():
    """External circuits with two coils for tests."""
    from jax_frc.circuits import (
        CircuitParams,
        CoilGeometry,
        CircuitDriver,
        ExternalCircuit,
        ExternalCircuits,
    )

    coil1 = CoilGeometry(z_center=0.0, radius=0.6, length=0.5, n_turns=50)
    coil2 = CoilGeometry(z_center=0.5, radius=0.6, length=0.5, n_turns=50)

    driver = CircuitDriver(mode="voltage", waveform=lambda t: 0.0)

    circuit1 = ExternalCircuit(
        name="coil1",
        coil=coil1,
        params=CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.1]),
            C=jnp.array([jnp.inf]),
        ),
        driver=driver,
    )
    circuit2 = ExternalCircuit(
        name="coil2",
        coil=coil2,
        params=CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.1]),
            C=jnp.array([jnp.inf]),
        ),
        driver=driver,
    )

    return ExternalCircuits(circuits=[circuit1, circuit2])


class TestFluxCouplingCreation:
    """Tests for FluxCoupling instantiation."""

    def test_import(self):
        """FluxCoupling can be imported."""
        from jax_frc.circuits.coupling import FluxCoupling

    def test_creation(self):
        """FluxCoupling can be instantiated."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()
        assert coupling is not None


class TestPlasmaToCoils:
    """Tests for plasma_to_coils method."""

    def test_plasma_to_coils_returns_correct_shapes(
        self, geometry, pickup_array, external_circuits
    ):
        """plasma_to_coils returns correct shapes."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        # Uniform Bz field
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B, geometry, pickup_array, external_circuits
        )

        assert Psi_pickup.shape == (3,)  # 3 pickup coils
        assert Psi_external.shape == (2,)  # 2 external circuits

    def test_plasma_to_coils_uniform_field(
        self, geometry, pickup_array, external_circuits
    ):
        """plasma_to_coils computes expected flux in uniform field."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        # Uniform Bz = 1 T field
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B, geometry, pickup_array, external_circuits
        )

        # All pickup coils should see similar flux (same radius, uniform field)
        assert jnp.allclose(Psi_pickup[0], Psi_pickup[1], rtol=0.1)
        assert jnp.allclose(Psi_pickup[1], Psi_pickup[2], rtol=0.1)

        # External coils at larger radius should have flux
        assert jnp.all(Psi_external > 0)

    def test_plasma_to_coils_nonuniform_field(
        self, geometry, pickup_array, external_circuits
    ):
        """plasma_to_coils correctly captures spatial variation."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        # Bz varies with z: stronger at z > 0
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        z_grid = geometry.z_grid
        Bz = jnp.where(z_grid > 0, 2.0, 1.0)
        B = B.at[:, :, 2].set(Bz)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B, geometry, pickup_array, external_circuits
        )

        # Coil at z=0.5 should see ~2x the flux of coil at z=-0.5
        assert Psi_pickup[2] > Psi_pickup[0]
        assert jnp.isclose(Psi_pickup[2] / Psi_pickup[0], 2.0, rtol=0.2)

    def test_plasma_to_coils_zero_field(
        self, geometry, pickup_array, external_circuits
    ):
        """plasma_to_coils returns zero flux for zero field."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        B = jnp.zeros((geometry.nr, geometry.nz, 3))

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B, geometry, pickup_array, external_circuits
        )

        assert jnp.allclose(Psi_pickup, 0.0)
        assert jnp.allclose(Psi_external, 0.0)


class TestCoilsToPlasma:
    """Tests for coils_to_plasma method."""

    def test_coils_to_plasma_returns_correct_shape(self, geometry, external_circuits):
        """coils_to_plasma returns correct shape."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        I_external = jnp.array([100.0, 100.0])

        B_coils = coupling.coils_to_plasma(I_external, external_circuits, geometry)

        assert B_coils.shape == (geometry.nr, geometry.nz, 3)

    def test_coils_to_plasma_zero_current(self, geometry, external_circuits):
        """coils_to_plasma returns zero field for zero current."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        I_external = jnp.array([0.0, 0.0])

        B_coils = coupling.coils_to_plasma(I_external, external_circuits, geometry)

        assert jnp.allclose(B_coils, 0.0)

    def test_coils_to_plasma_nonzero_current(self, geometry, external_circuits):
        """coils_to_plasma returns nonzero field for nonzero current."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        I_external = jnp.array([100.0, 100.0])

        B_coils = coupling.coils_to_plasma(I_external, external_circuits, geometry)

        # Should have nonzero Bz component
        assert jnp.any(B_coils[:, :, 2] != 0.0)

    def test_coils_to_plasma_current_scaling(self, geometry, external_circuits):
        """coils_to_plasma field scales linearly with current."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        I1 = jnp.array([100.0, 100.0])
        I2 = jnp.array([200.0, 200.0])

        B1 = coupling.coils_to_plasma(I1, external_circuits, geometry)
        B2 = coupling.coils_to_plasma(I2, external_circuits, geometry)

        # Field should scale linearly
        assert jnp.allclose(B2, 2 * B1, rtol=1e-5)


class TestJITCompatibility:
    """Tests for JAX JIT compatibility."""

    def test_plasma_to_coils_jit(self, geometry, pickup_array, external_circuits):
        """plasma_to_coils is JIT-compatible."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        # Should compile and run without errors
        @jit
        def compute_flux(B_field):
            return coupling.plasma_to_coils(
                B_field, geometry, pickup_array, external_circuits
            )

        Psi_pickup, Psi_external = compute_flux(B)

        assert Psi_pickup.shape == (3,)
        assert Psi_external.shape == (2,)

    def test_coils_to_plasma_jit(self, geometry, external_circuits):
        """coils_to_plasma is JIT-compatible."""
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()

        I_external = jnp.array([100.0, 100.0])

        # Should compile and run without errors
        @jit
        def compute_field(I):
            return coupling.coils_to_plasma(I, external_circuits, geometry)

        B_coils = compute_field(I_external)

        assert B_coils.shape == (geometry.nr, geometry.nz, 3)


class TestEmptyCircuits:
    """Tests for edge cases with empty circuit arrays."""

    def test_empty_external_circuits(self, geometry, pickup_array):
        """Handles empty external circuits gracefully."""
        from jax_frc.circuits import ExternalCircuits
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()
        empty_external = ExternalCircuits(circuits=[])

        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B, geometry, pickup_array, empty_external
        )

        assert Psi_pickup.shape == (3,)
        assert Psi_external.shape == (0,)

    def test_coils_to_plasma_empty_external(self, geometry):
        """coils_to_plasma handles empty external circuits."""
        from jax_frc.circuits import ExternalCircuits
        from jax_frc.circuits.coupling import FluxCoupling

        coupling = FluxCoupling()
        empty_external = ExternalCircuits(circuits=[])

        I_external = jnp.array([])

        B_coils = coupling.coils_to_plasma(I_external, empty_external, geometry)

        assert B_coils.shape == (geometry.nr, geometry.nz, 3)
        assert jnp.allclose(B_coils, 0.0)
