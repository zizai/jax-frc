"""Tests for burning plasma model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16, nz=32,
        r_min=0.1, r_max=0.5,
        z_min=-1.0, z_max=1.0,
    )


class TestBurningPlasmaState:
    """Tests for BurningPlasmaState dataclass."""

    def test_state_creation(self, geometry):
        """Can create BurningPlasmaState."""
        from jax_frc.models.burning_plasma import BurningPlasmaState
        from jax_frc.core.state import State
        from jax_frc.burn import SpeciesState, ReactionRates, PowerSources, ConversionState

        mhd = State.zeros(geometry.nr, geometry.nz)
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )
        rates = ReactionRates(
            DT=jnp.zeros((geometry.nr, geometry.nz)),
            DD_T=jnp.zeros((geometry.nr, geometry.nz)),
            DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
            DHE3=jnp.zeros((geometry.nr, geometry.nz)),
        )
        power = PowerSources(
            P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
            P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
            P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
            P_charged=jnp.zeros((geometry.nr, geometry.nz)),
        )
        conversion = ConversionState(P_electric=0.0, V_induced=0.0, dPsi_dt=0.0)

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=rates,
            power=power,
            conversion=conversion,
        )

        assert state.mhd is not None
        assert state.species.n_D.shape == (geometry.nr, geometry.nz)

    def test_state_is_pytree(self, geometry):
        """BurningPlasmaState works with JAX transformations."""
        from jax_frc.models.burning_plasma import BurningPlasmaState
        from jax_frc.core.state import State
        from jax_frc.burn import SpeciesState, ReactionRates, PowerSources, ConversionState

        mhd = State.zeros(geometry.nr, geometry.nz)
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )
        rates = ReactionRates(
            DT=jnp.zeros((geometry.nr, geometry.nz)),
            DD_T=jnp.zeros((geometry.nr, geometry.nz)),
            DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
            DHE3=jnp.zeros((geometry.nr, geometry.nz)),
        )
        power = PowerSources(
            P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
            P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
            P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
            P_charged=jnp.zeros((geometry.nr, geometry.nz)),
        )
        conversion = ConversionState(P_electric=0.0, V_induced=0.0, dPsi_dt=0.0)

        state = BurningPlasmaState(
            mhd=mhd, species=species, rates=rates, power=power, conversion=conversion,
        )

        @jax.jit
        def get_fusion_power(s):
            return s.power.P_fusion

        result = get_fusion_power(state)
        assert result.shape == (geometry.nr, geometry.nz)
