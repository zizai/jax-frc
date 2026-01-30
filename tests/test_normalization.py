"""Tests for normalization utilities."""

import jax.numpy as jnp
import pytest

from jax_frc.core.state import State
from jax_frc.units.normalization import (
    NormScales,
    to_dimless_state,
    to_physical_state,
    scale_eta_nu,
)


def test_state_round_trip():
    scales = NormScales(L0=1.0, rho0=2.0, B0=3.0)
    state = State.zeros(4, 2, 4)
    state = state.replace(
        n=state.n + 1.0,
        p=state.p + 2.0,
        v=jnp.ones_like(state.B),
    )
    dimless = to_dimless_state(state, scales)
    physical = to_physical_state(dimless, scales)

    assert jnp.allclose(physical.n, state.n)
    assert jnp.allclose(physical.p, state.p)
    assert jnp.allclose(physical.v, state.v)


def test_eta_nu_scaling():
    scales = NormScales(L0=2.0, rho0=1.0, B0=4.0)
    eta_star, nu_star = scale_eta_nu(eta=0.5, nu=0.25, scales=scales)
    v0 = scales.v0

    assert eta_star == pytest.approx(0.5 / (scales.L0 * v0))
    assert nu_star == pytest.approx(0.25 / (scales.L0 * v0))
