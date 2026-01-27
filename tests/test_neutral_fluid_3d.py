"""Tests for 3D Cartesian neutral fluid model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.constants import QE, MI
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState, GAMMA


class TestNeutralState3D:
    """Tests for NeutralState with 3D arrays."""

    def test_state_creation(self):
        """Test creating neutral state with 3D arrays."""
        state = NeutralState(
            rho_n=jnp.ones((4, 6, 8)) * 1e-6,
            mom_n=jnp.zeros((4, 6, 8, 3)),
            E_n=jnp.ones((4, 6, 8)) * 1e-3,
        )
        assert state.rho_n.shape == (4, 6, 8)
        assert state.mom_n.shape == (4, 6, 8, 3)
        assert state.E_n.shape == (4, 6, 8)

    def test_derived_properties(self):
        """Test velocity and pressure calculations."""
        rho = jnp.ones((4, 4, 4)) * 1e-6
        mom = jnp.zeros((4, 4, 4, 3))
        mom = mom.at[..., 0].set(1e-6)  # v_x = 1 m/s
        E = jnp.ones((4, 4, 4)) * 1e-3

        state = NeutralState(rho_n=rho, mom_n=mom, E_n=E)
        assert state.v_n.shape == (4, 4, 4, 3)
        assert state.p_n.shape == (4, 4, 4)

    def test_velocity_calculation(self):
        """Test v_n = mom_n / rho_n for 3D arrays."""
        rho = jnp.ones((4, 4, 4)) * 1e-6
        mom = jnp.zeros((4, 4, 4, 3))
        mom = mom.at[..., 0].set(1e-6 * 100)  # v_x = 100 m/s
        mom = mom.at[..., 1].set(1e-6 * 200)  # v_y = 200 m/s
        mom = mom.at[..., 2].set(1e-6 * 300)  # v_z = 300 m/s
        E = jnp.ones((4, 4, 4)) * 1e3

        state = NeutralState(rho_n=rho, mom_n=mom, E_n=E)
        assert jnp.allclose(state.v_n[..., 0], 100.0)
        assert jnp.allclose(state.v_n[..., 1], 200.0)
        assert jnp.allclose(state.v_n[..., 2], 300.0)

    def test_pressure_calculation(self):
        """Test pressure from ideal gas EOS for 3D arrays."""
        rho = jnp.ones((4, 4, 4)) * MI * 1e19  # n = 1e19 m^-3
        mom = jnp.zeros((4, 4, 4, 3))  # Stationary
        # E_n = p / (gamma - 1) for stationary gas
        p_target = 1e19 * 10 * QE  # n * T where T = 10 eV
        E = jnp.ones((4, 4, 4)) * p_target / (GAMMA - 1)

        state = NeutralState(rho_n=rho, mom_n=mom, E_n=E)
        assert jnp.allclose(state.p_n, p_target, rtol=1e-5)

    def test_state_is_pytree(self):
        """NeutralState works with JAX transformations in 3D."""
        state = NeutralState(
            rho_n=jnp.ones((4, 4, 4)) * 1e-6,
            mom_n=jnp.zeros((4, 4, 4, 3)),
            E_n=jnp.ones((4, 4, 4)) * 1e3,
        )

        @jax.jit
        def get_density(s):
            return s.rho_n

        result = get_density(state)
        assert result.shape == (4, 4, 4)


class TestNeutralFluid3D:
    """Tests for NeutralFluid model with 3D Cartesian geometry."""

    def test_flux_divergence_shape(self):
        """Test flux divergence returns correct shapes."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.zeros((8, 8, 8, 3)),
            E_n=jnp.ones((8, 8, 8)) * 1e-3,
        )

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geom)
        assert d_rho.shape == (8, 8, 8)
        assert d_mom.shape == (8, 8, 8, 3)
        assert d_E.shape == (8, 8, 8)

    def test_flux_divergence_asymmetric_shape(self):
        """Test flux divergence with asymmetric grid."""
        model = NeutralFluid()
        geom = Geometry(nx=6, ny=8, nz=10)
        state = NeutralState(
            rho_n=jnp.ones((6, 8, 10)) * 1e-6,
            mom_n=jnp.zeros((6, 8, 10, 3)),
            E_n=jnp.ones((6, 8, 10)) * 1e-3,
        )

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geom)
        assert d_rho.shape == (6, 8, 10)
        assert d_mom.shape == (6, 8, 10, 3)
        assert d_E.shape == (6, 8, 10)

    def test_uniform_state_zero_flux_divergence(self):
        """Uniform stationary state has ~zero flux divergence."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)

        # Uniform stationary state
        p = 1e3
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.zeros((8, 8, 8, 3)),
            E_n=jnp.ones((8, 8, 8)) * p / (GAMMA - 1),
        )

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geom)

        # For uniform state with periodic BC, flux divergence should be near zero
        assert jnp.max(jnp.abs(d_rho)) < 1e-10
        assert jnp.max(jnp.abs(d_E)) < 1e-10

    def test_compute_flux_dir_method_exists(self):
        """Test that _compute_flux_dir method exists."""
        model = NeutralFluid()
        assert hasattr(model, '_compute_flux_dir')

    def test_flux_divergence_jittable(self):
        """Test flux divergence can be JIT compiled."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.zeros((8, 8, 8, 3)),
            E_n=jnp.ones((8, 8, 8)) * 1e-3,
        )

        # Geometry is not a pytree, so we need to use partial or closure
        def compute_flux(s):
            return model.compute_flux_divergence(s, geom)

        compute_flux_jit = jax.jit(compute_flux)
        d_rho, d_mom, d_E = compute_flux_jit(state)
        assert d_rho.shape == (8, 8, 8)


class TestNeutralBoundaryConditions3D:
    """Tests for 3D boundary conditions."""

    def test_apply_boundary_conditions_exists(self):
        """Method exists on NeutralFluid."""
        model = NeutralFluid()
        assert hasattr(model, 'apply_boundary_conditions')

    def test_periodic_bc_returns_state_unchanged(self):
        """For periodic BC, state should be returned unchanged."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.ones((8, 8, 8, 3)) * 1e-6 * 100,
            E_n=jnp.ones((8, 8, 8)) * 1e-3,
        )

        state_bc = model.apply_boundary_conditions(state, geom, bc_type="periodic")

        # State should be unchanged for periodic BC
        assert jnp.allclose(state_bc.rho_n, state.rho_n)
        assert jnp.allclose(state_bc.mom_n, state.mom_n)
        assert jnp.allclose(state_bc.E_n, state.E_n)

    def test_boundary_conditions_jittable(self):
        """Test boundary conditions can be JIT compiled."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.zeros((8, 8, 8, 3)),
            E_n=jnp.ones((8, 8, 8)) * 1e-3,
        )

        # Geometry is not a pytree, use closure
        def apply_bc(s):
            return model.apply_boundary_conditions(s, geom, bc_type="periodic")

        apply_bc_jit = jax.jit(apply_bc)
        state_bc = apply_bc_jit(state)
        assert state_bc.rho_n.shape == (8, 8, 8)
