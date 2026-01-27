# tests/test_cx_equilibration.py
"""Validation test: CX momentum relaxation with analytic solution."""

import jax.numpy as jnp
import jax.lax as lax
from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE
from tests.utils.cartesian import make_geometry


def test_cx_velocity_relaxation():
    """CX friction equilibrates plasma-neutral velocities exponentially.

    Setup: Stationary plasma, drifting neutrals
    Physics: The relative velocity Delta_v = v_n - v_i relaxes exponentially

    For plasma:  d(v_i)/dt = nu_cx * (v_n - v_i)  where nu_cx = n_n * sigma * v_th
    For neutrals: d(v_n)/dt = -n_i/n_n * nu_cx * (v_n - v_i)

    Combined: d(Delta_v)/dt = -(1 + n_i/n_n) * nu_cx * Delta_v
                            = -(n_n + n_i) * sigma * v_th * Delta_v

    So tau_eff = 1 / [(n_n + n_i) * sigma * v_th]
    """
    # Physical parameters
    n_i = 1e19  # m^-3
    n_e = n_i   # Quasi-neutrality
    T_i = 100 * QE  # 100 eV in Joules
    n_n = 1e18  # m^-3 (lower than plasma)
    v_n0 = 1000.0  # Initial neutral velocity [m/s]

    # CX timescale - must account for momentum exchange in BOTH directions
    # The relative velocity decays with rate (n_n + n_i) * sigma * v_th
    sigma_cx = 3e-19  # m^2
    v_thermal = jnp.sqrt(8 * T_i / (jnp.pi * MI))
    # Effective timescale for relative velocity relaxation
    tau_eff = 1.0 / ((n_n + n_i) * sigma_cx * v_thermal)

    geometry = make_geometry(nx=4, ny=1, nz=4, extent=0.1)

    # Initial state: stationary plasma, moving neutrals
    plasma = State.zeros(4, 1, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 1, 4)) * n_e,
        Te=jnp.ones((4, 1, 4)) * T_i,
        p=jnp.ones((4, 1, 4)) * n_e * T_i * 2,
        v=jnp.zeros((4, 1, 4, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 1, 4)) * n_n * MI,
        mom_n=jnp.zeros((4, 1, 4, 3)).at[:, :, :, 2].set(n_n * MI * v_n0),
        E_n=jnp.ones((4, 1, 4)) * 0.5 * n_n * MI * v_n0**2
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))

    # Time evolution of velocity difference
    dt = tau_eff / 100  # Small timesteps for accuracy
    n_steps = 200

    def step(carry, _):
        plasma_v, neutral_v_z, neutral_rho = carry

        # Compute source rates
        # Create temporary states for coupling
        temp_plasma = plasma.replace(
            v=jnp.zeros((4, 1, 4, 3)).at[:, :, :, 2].set(plasma_v)
        )
        temp_neutral = NeutralState(
            rho_n=neutral_rho,
            mom_n=jnp.zeros((4, 1, 4, 3)).at[:, :, :, 2].set(neutral_rho * neutral_v_z),
            E_n=jnp.ones((4, 1, 4)) * 100.0
        )

        plasma_src, neutral_src = coupling.compute_sources(temp_plasma, temp_neutral, geometry)

        # R_cx is momentum transfer to plasma (positive if neutrals faster)
        R_cx_z = plasma_src.momentum[:, :, :, 2]

        # Update velocities: dv/dt = R_cx / (rho)
        # For plasma: rho = n_i * MI
        # For neutrals: -R_cx / rho_n
        dv_plasma = R_cx_z / (n_i * MI) * dt
        dv_neutral = -R_cx_z / neutral_rho * dt

        new_plasma_v = plasma_v + dv_plasma[0, 0, 0]  # Uniform, take one cell
        new_neutral_v = neutral_v_z + dv_neutral[0, 0, 0]

        delta_v = new_neutral_v - new_plasma_v
        return (new_plasma_v, new_neutral_v, neutral_rho), delta_v

    neutral_rho = jnp.ones((4, 1, 4)) * n_n * MI
    init_carry = (0.0, v_n0, neutral_rho)
    _, delta_v_history = lax.scan(step, init_carry, None, length=n_steps)

    # Compare to analytic: delta_v(t) = v_n0 * exp(-t/tau_eff)
    times = jnp.arange(n_steps) * dt
    analytic = v_n0 * jnp.exp(-times / tau_eff)

    # Check exponential decay (within 20% due to approximations)
    # At t = tau_eff, should be at ~37% of initial
    idx_tau = int(tau_eff / dt)
    if idx_tau < n_steps:
        numerical_ratio = delta_v_history[idx_tau] / v_n0
        expected_ratio = jnp.exp(-1)  # ~0.368
        assert jnp.abs(numerical_ratio - expected_ratio) < 0.2, \
            f"At t=tau_eff: numerical {numerical_ratio:.3f} vs expected {expected_ratio:.3f}"


def test_cx_momentum_conservation():
    """Total momentum conserved during CX exchange."""
    n_i = 1e19
    n_e = n_i
    T_i = 100 * QE
    n_n = 1e19
    v_n0 = 1000.0

    geometry = make_geometry(nx=4, ny=1, nz=4, extent=0.1)

    plasma = State.zeros(4, 1, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 1, 4)) * n_e,
        Te=jnp.ones((4, 1, 4)) * T_i,
        p=jnp.ones((4, 1, 4)) * n_e * T_i * 2,
        v=jnp.zeros((4, 1, 4, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 1, 4)) * n_n * MI,
        mom_n=jnp.zeros((4, 1, 4, 3)).at[:, :, :, 2].set(n_n * MI * v_n0),
        E_n=jnp.ones((4, 1, 4)) * 100.0
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Total momentum source should be zero
    total_mom = plasma_src.momentum + neutral_src.momentum
    assert jnp.allclose(total_mom, 0.0, atol=1e-25), "Momentum not conserved"
