# tests/test_coupled_model.py
"""Tests for CoupledModel composition wrapper."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState, NeutralFluid
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.models.coupled import CoupledState, CoupledModel, CoupledModelConfig
from jax_frc.constants import QE
from tests.utils.cartesian import make_geometry


def test_coupled_model_explicit_rhs():
    """CoupledModel.explicit_rhs combines plasma and neutral advection."""
    plasma_model = ResistiveMHD(eta=1e-6)
    neutral_model = NeutralFluid()
    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    config = CoupledModelConfig(source_subcycles=5)

    model = CoupledModel(plasma_model, neutral_model, coupling, config)

    geometry = make_geometry(nx=8, ny=1, nz=8, extent=0.5)

    plasma = State.zeros(8, 1, 8)
    B_init = jnp.zeros((8, 1, 8, 3))
    B_init = B_init.at[:, :, :, 2].set(0.1)
    plasma = plasma.replace(
        B=B_init,
        n=jnp.ones((8, 1, 8)) * 1e19,
        p=jnp.ones((8, 1, 8)) * 1e19 * 100 * QE * 2,
        v=jnp.zeros((8, 1, 8, 3)),
        Te=jnp.ones((8, 1, 8)) * 100 * QE
    )
    neutral = NeutralState(
        rho_n=jnp.ones((8, 1, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 1, 8, 3)),
        E_n=jnp.ones((8, 1, 8)) * 100.0
    )
    state = CoupledState(plasma=plasma, neutral=neutral)

    rhs = model.explicit_rhs(state, geometry, t=0.0)

    assert isinstance(rhs, CoupledState)
    assert rhs.plasma.B.shape == (8, 1, 8, 3)
    assert rhs.neutral.rho_n.shape == (8, 1, 8)


def test_coupled_model_source_rhs():
    """CoupledModel.source_rhs computes atomic source terms."""
    plasma_model = ResistiveMHD(eta=1e-6)
    neutral_model = NeutralFluid()
    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    config = CoupledModelConfig(source_subcycles=5)

    model = CoupledModel(plasma_model, neutral_model, coupling, config)

    geometry = make_geometry(nx=8, ny=1, nz=8, extent=0.5)

    plasma = State.zeros(8, 1, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 1, 8)) * 1e19,
        p=jnp.ones((8, 1, 8)) * 1e19 * 100 * QE * 2,
        v=jnp.zeros((8, 1, 8, 3)),
        Te=jnp.ones((8, 1, 8)) * 100 * QE
    )
    neutral = NeutralState(
        rho_n=jnp.ones((8, 1, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 1, 8, 3)),
        E_n=jnp.ones((8, 1, 8)) * 100.0
    )
    state = CoupledState(plasma=plasma, neutral=neutral)

    rhs = model.source_rhs(state, geometry, t=0.0)

    assert isinstance(rhs, CoupledState)
    # Ionization should create positive plasma mass source
    assert jnp.any(rhs.plasma.n != 0)
