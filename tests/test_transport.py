"""Tests for transport model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def geometry():
    return Geometry(
        nx=16,
        ny=4,
        nz=32,
        x_min=0.1,
        x_max=0.5,
        y_min=0.0,
        y_max=2 * jnp.pi,
        z_min=-1.0,
        z_max=1.0,
        bc_x="neumann",
        bc_y="periodic",
        bc_z="neumann",
    )


class TestTransportModel:
    """Tests for anomalous transport."""

    def test_transport_model_creation(self):
        """Can create TransportModel with diffusivities."""
        from jax_frc.transport import TransportModel
        transport = TransportModel(
            D_particle=1.0,
            chi_e=2.0,
            chi_i=1.0,
        )
        assert transport.D_particle == 1.0
        assert transport.chi_e == 2.0

    def test_particle_flux_diffusive(self, geometry):
        """Particle flux Gamma = -D * grad(n)."""
        from jax_frc.transport import TransportModel
        transport = TransportModel(D_particle=1.0, chi_e=1.0, chi_i=1.0)

        # Linear density profile: n(r) = n0 * (1 - r/r_max)
        x_mid = geometry.ny // 2
        r = geometry.x_grid[:, x_mid, :]
        n = 1e20 * (1 - r / geometry.x_max)

        Gamma_r, Gamma_z = transport.particle_flux(n, geometry)

        # dn/dr = -n0/r_max, so Gamma_r = D * n0/r_max
        # Interior points should have positive radial flux (outward)
        assert jnp.all(Gamma_r[2:-2, 2:-2] > 0)

    def test_energy_flux_diffusive(self, geometry):
        """Energy flux q = -n * chi * grad(T)."""
        from jax_frc.transport import TransportModel
        transport = TransportModel(D_particle=1.0, chi_e=5.0, chi_i=5.0)

        n = jnp.ones((geometry.nx, geometry.nz)) * 1e20
        # Linear temperature profile
        x_mid = geometry.ny // 2
        r = geometry.x_grid[:, x_mid, :]
        T = 10.0 * (1 - r / geometry.x_max)  # keV

        q_r, q_z = transport.energy_flux(n, T, geometry)

        # Should have outward heat flux where dT/dr < 0
        assert jnp.all(q_r[2:-2, 2:-2] > 0)

    def test_zero_flux_uniform_profiles(self, geometry):
        """Zero flux for uniform profiles."""
        from jax_frc.transport import TransportModel
        transport = TransportModel(D_particle=1.0, chi_e=1.0, chi_i=1.0)

        n = jnp.ones((geometry.nx, geometry.nz)) * 1e20
        T = jnp.ones((geometry.nx, geometry.nz)) * 10.0

        Gamma_r, Gamma_z = transport.particle_flux(n, geometry)
        q_r, q_z = transport.energy_flux(n, T, geometry)

        # Interior should have near-zero flux
        assert jnp.allclose(Gamma_r[2:-2, 2:-2], 0, atol=1e10)
        assert jnp.allclose(q_r[2:-2, 2:-2], 0, atol=1e10)
