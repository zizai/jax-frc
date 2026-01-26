"""Tests for neutral fluid model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.constants import QE, MI


class TestNeutralState:
    """Tests for NeutralState dataclass."""

    def test_neutral_state_importable(self):
        """NeutralState is importable."""
        from jax_frc.models.neutral_fluid import NeutralState
        assert NeutralState is not None

    def test_neutral_state_creation(self):
        """Can create NeutralState with required fields."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert state.rho_n.shape == (16, 32)
        assert state.mom_n.shape == (16, 32, 3)
        assert state.E_n.shape == (16, 32)

    def test_neutral_state_velocity_property(self):
        """v_n = mom_n / rho_n."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.ones((16, 32, 3)) * 1e-6 * 1000  # 1000 m/s
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.v_n, 1000.0)

    def test_neutral_state_pressure_property(self):
        """p_n from ideal gas EOS."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * MI * 1e19  # n = 1e19 m^-3
        mom_n = jnp.zeros((16, 32, 3))  # Stationary
        # E_n = p / (gamma - 1) for stationary gas
        gamma = 5/3
        p_target = 1e19 * 10 * QE  # n * T where T = 10 eV
        E_n = p_target / (gamma - 1)
        E_n = jnp.ones((16, 32)) * E_n
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.p_n, p_target, rtol=1e-5)

    def test_neutral_state_is_pytree(self):
        """NeutralState works with JAX transformations."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        # Should be able to use in JIT
        @jax.jit
        def get_density(s):
            return s.rho_n

        result = get_density(state)
        assert result.shape == (16, 32)


class TestEulerFlux:
    """Tests for Euler flux computations."""

    def test_euler_flux_exists(self):
        """Function is importable."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        assert callable(euler_flux_1d)

    def test_euler_flux_mass(self):
        """Mass flux = rho * v."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_rho, rho * v)

    def test_euler_flux_momentum(self):
        """Momentum flux = rho * v² + p."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_mom, rho * v**2 + p)

    def test_euler_flux_energy(self):
        """Energy flux = (E + p) * v."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_E, (E + p) * v)


class TestHLLEFlux:
    """Tests for HLLE approximate Riemann solver."""

    def test_hlle_flux_exists(self):
        """Function is importable."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d
        assert callable(hlle_flux_1d)

    def test_hlle_flux_uniform_state(self):
        """HLLE returns physical flux for uniform state."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d, euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2

        # Same state on left and right
        F_hlle = hlle_flux_1d(rho, rho, v, v, p, p, E, E)
        F_exact = euler_flux_1d(rho, v, p, E)

        assert jnp.allclose(F_hlle[0], F_exact[0], rtol=1e-5)
        assert jnp.allclose(F_hlle[1], F_exact[1], rtol=1e-5)
        assert jnp.allclose(F_hlle[2], F_exact[2], rtol=1e-5)

    def test_hlle_flux_supersonic_right(self):
        """For supersonic flow to right, use left flux."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d, euler_flux_1d
        # Supersonic flow: v >> c_s
        rho = 1.0
        v = 1000.0  # Much faster than sound speed ~300 m/s
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2

        # Slight perturbation on right
        F_hlle = hlle_flux_1d(rho, rho*1.01, v, v, p, p*1.01, E, E*1.01)
        F_left = euler_flux_1d(rho, v, p, E)

        # Should be close to left flux
        assert jnp.allclose(F_hlle[0], F_left[0], rtol=0.1)


class TestNeutralFluid:
    """Tests for NeutralFluid model class."""

    def test_neutral_fluid_importable(self):
        """NeutralFluid is importable."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        assert NeutralFluid is not None

    def test_neutral_fluid_creation(self):
        """Can create NeutralFluid instance."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        model = NeutralFluid(gamma=5/3)
        assert model.gamma == 5/3

    def test_compute_flux_divergence_shape(self):
        """Flux divergence returns correct shapes."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
        from jax_frc.core.geometry import Geometry

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(
            coord_system="cylindrical",
            nr=nr, nz=nz,
            r_min=0.1, r_max=1.0,
            z_min=0.0, z_max=2.0
        )

        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.zeros((nr, nz, 3))
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geometry)

        assert d_rho.shape == (nr, nz)
        assert d_mom.shape == (nr, nz, 3)
        assert d_E.shape == (nr, nz)

    def test_uniform_state_zero_flux_divergence(self):
        """Uniform stationary state has ~zero flux divergence."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
        from jax_frc.core.geometry import Geometry

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(
            coord_system="cylindrical",
            nr=nr, nz=nz,
            r_min=0.1, r_max=1.0,
            z_min=0.0, z_max=2.0
        )

        # Uniform stationary state
        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.zeros((nr, nz, 3))
        p_n = 1e3
        E_n = jnp.ones((nr, nz)) * p_n / (5/3 - 1)
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geometry)

        # Interior should be near zero (boundaries may have edge effects)
        assert jnp.max(jnp.abs(d_rho[2:-2, 2:-2])) < 1e-10


class TestNeutralBoundaryConditions:
    """Tests for neutral boundary conditions."""

    def test_apply_boundary_conditions_exists(self):
        """Method exists on NeutralFluid."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        model = NeutralFluid()
        assert hasattr(model, 'apply_boundary_conditions')

    def test_reflecting_bc_reverses_normal_velocity(self):
        """Reflecting BC reverses velocity normal to wall."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
        from jax_frc.core.geometry import Geometry

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(
            coord_system="cylindrical",
            nr=nr, nz=nz,
            r_min=0.1, r_max=1.0,
            z_min=0.0, z_max=2.0
        )

        rho_n = jnp.ones((nr, nz)) * 1e-6
        # Velocity pointing outward at outer r boundary
        mom_n = jnp.zeros((nr, nz, 3))
        mom_n = mom_n.at[-1, :, 0].set(1e-6 * 100)  # v_r = 100 at outer wall
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        state_bc = model.apply_boundary_conditions(state, geometry, bc_type="reflecting")

        # v_r should be zero or reversed at outer boundary
        assert jnp.all(state_bc.mom_n[-1, :, 0] <= 0)

    def test_axis_symmetry(self):
        """Axis (r=0) has correct symmetry."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
        from jax_frc.core.geometry import Geometry

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(
            coord_system="cylindrical",
            nr=nr, nz=nz,
            r_min=0.1, r_max=1.0,
            z_min=0.0, z_max=2.0
        )

        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.ones((nr, nz, 3)) * 1e-6 * 100
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        state_bc = model.apply_boundary_conditions(state, geometry)

        # v_r = 0 at axis
        assert jnp.allclose(state_bc.mom_n[0, :, 0], 0, atol=1e-20)
