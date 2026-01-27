"""Validation test: Alfven wave and resistive damping.

This test validates resistive damping of magnetic perturbations in the 3D
Cartesian ResistiveMHD model. For a sinusoidal perturbation B_y = dB*sin(k*z),
the resistive damping rate is gamma = eta * k^2 / mu_0.

Note: The ResistiveMHD model with v=0 handles resistive diffusion only
(dB/dt = -curl(eta*J)). True Alfven wave propagation requires the v x B
term in Ohm's law with non-zero velocity.
"""

import jax.numpy as jnp
import pytest
from jax import lax
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.constants import MU0


def rk4_step(state, dt, model, geometry):
    """RK4 step for 3D State.

    Manual RK4 implementation that works with the 3D State class,
    evolving only the B field through the ResistiveMHD model.
    """
    k1 = model.compute_rhs(state, geometry)
    state_k2 = state.replace(B=state.B + 0.5 * dt * k1.B)
    k2 = model.compute_rhs(state_k2, geometry)
    state_k3 = state.replace(B=state.B + 0.5 * dt * k2.B)
    k3 = model.compute_rhs(state_k3, geometry)
    state_k4 = state.replace(B=state.B + dt * k3.B)
    k4 = model.compute_rhs(state_k4, geometry)
    B_new = state.B + (dt / 6) * (k1.B + 2 * k2.B + 2 * k3.B + k4.B)
    return state.replace(B=B_new)


class TestResistiveDamping:
    """Test resistive damping of magnetic perturbations."""

    @pytest.mark.slow
    def test_sinusoidal_damping(self):
        """Sinusoidal mode should decay at rate gamma = k^2 * eta / mu_0.

        For a perturbation B_y = dB * sin(k*z), the resistive diffusion
        equation gives exponential decay: B_y(t) = B_y(0) * exp(-gamma*t)
        where gamma = eta * k^2 / mu_0.
        """
        # Parameters
        eta = 1e-3  # Resistivity [Ohm*m]
        B0 = 1.0  # Background field strength [T]
        dB = 0.01  # Perturbation amplitude [T]

        # Grid - periodic in all directions for sinusoidal mode
        Lz = 1.0
        nz = 64
        geom = Geometry(
            nx=4,
            ny=4,
            nz=nz,
            x_min=0.0,
            x_max=0.1,
            y_min=0.0,
            y_max=0.1,
            z_min=0.0,
            z_max=Lz,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )

        # Initial condition: B_z = B0 (uniform), B_y = dB * sin(k*z)
        k = 2 * jnp.pi / Lz  # Wavenumber for one complete wavelength
        z = geom.z_grid
        B_init = jnp.zeros((4, 4, nz, 3))
        B_init = B_init.at[..., 2].set(B0)  # Uniform B_z
        B_init = B_init.at[..., 1].set(dB * jnp.sin(k * z))  # Sinusoidal B_y

        state = State.zeros(4, 4, nz)
        state = state.replace(B=B_init, n=jnp.ones((4, 4, nz)) * 1e19)

        # Analytic decay rate for sinusoidal mode
        gamma = eta * k**2 / MU0  # Decay rate [1/s]
        t_decay = 1.0 / gamma  # e-folding time [s]

        # Evolve for fraction of decay time to see measurable decay
        t_final = 0.5 * t_decay
        model = ResistiveMHD(eta=eta)

        # Compute stable timestep from CFL condition
        dx_min = min(geom.dx, geom.dy, geom.dz)
        diffusivity = eta / MU0
        dt_stable = 0.25 * dx_min**2 / diffusivity
        dt = min(dt_stable, 1e-6)  # Small stable timestep
        n_steps = int(t_final / dt)

        # Time evolution using lax.fori_loop for efficiency
        def body_fn(i, state):
            return rk4_step(state, dt, model, geom)

        state_final = lax.fori_loop(0, n_steps, body_fn, state)

        # Check amplitude decay
        By_init_amp = jnp.max(jnp.abs(B_init[..., 1]))
        By_final_amp = jnp.max(jnp.abs(state_final.B[..., 1]))

        # Expected decay: exp(-gamma * t_final) = exp(-0.5) approx 0.606
        actual_time = n_steps * dt
        expected_ratio = jnp.exp(-gamma * actual_time)
        actual_ratio = By_final_amp / By_init_amp

        # Allow 20% tolerance due to numerical discretization
        rel_error = jnp.abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_error < 0.2, (
            f"Decay ratio {float(actual_ratio):.4f} differs from expected "
            f"{float(expected_ratio):.4f} by more than 20% (rel_error={float(rel_error):.3f})"
        )

    @pytest.mark.slow
    def test_wavenumber_scaling(self):
        """Higher wavenumber modes should decay faster (gamma ~ k^2).

        For two sinusoidal modes with different wavenumbers k1 and k2,
        the decay rates should satisfy gamma2/gamma1 = (k2/k1)^2.
        """
        eta = 1e-4
        dB = 0.01

        def run_decay_test(nz, n_wavelengths):
            """Run decay test for given number of wavelengths."""
            Lz = 1.0
            geom = Geometry(
                nx=4,
                ny=4,
                nz=nz,
                x_min=0.0,
                x_max=0.1,
                y_min=0.0,
                y_max=0.1,
                z_min=0.0,
                z_max=Lz,
                bc_x="periodic",
                bc_y="periodic",
                bc_z="periodic",
            )

            k = 2 * jnp.pi * n_wavelengths / Lz
            z = geom.z_grid
            B_init = jnp.zeros((4, 4, nz, 3))
            B_init = B_init.at[..., 1].set(dB * jnp.sin(k * z))

            state = State.zeros(4, 4, nz)
            state = state.replace(B=B_init, n=jnp.ones((4, 4, nz)) * 1e19)

            model = ResistiveMHD(eta=eta)
            dx_min = min(geom.dx, geom.dy, geom.dz)
            diffusivity = eta / MU0
            dt = 0.1 * dx_min**2 / diffusivity  # Conservative timestep

            # Evolve for fixed time
            t_final = 1e-4
            n_steps = max(10, int(t_final / dt))

            def body_fn(i, state):
                return rk4_step(state, dt, model, geom)

            state_final = lax.fori_loop(0, n_steps, body_fn, state)

            By_init = jnp.max(jnp.abs(B_init[..., 1]))
            By_final = jnp.max(jnp.abs(state_final.B[..., 1]))
            return float(By_final / By_init), n_steps * dt

        # Test mode with 1 wavelength vs 2 wavelengths
        ratio1, t1 = run_decay_test(64, 1)
        ratio2, t2 = run_decay_test(64, 2)

        # Decay ratios: ratio = exp(-gamma*t), so log(ratio) = -gamma*t
        # For same evolution time, gamma2/gamma1 = log(ratio2)/log(ratio1)
        # Expected: (k2/k1)^2 = 4 (since k2 = 2*k1)
        gamma_ratio_actual = jnp.log(ratio2) / jnp.log(ratio1)
        gamma_ratio_expected = 4.0  # (2/1)^2

        # Allow 30% tolerance
        rel_error = jnp.abs(gamma_ratio_actual - gamma_ratio_expected) / gamma_ratio_expected
        assert rel_error < 0.3, (
            f"Wavenumber scaling: gamma ratio {float(gamma_ratio_actual):.2f} "
            f"differs from expected {gamma_ratio_expected:.2f}"
        )

    def test_uniform_field_stability(self):
        """Uniform magnetic field should remain constant (no diffusion)."""
        eta = 1e-3
        B0 = 1.0

        geom = Geometry(
            nx=8,
            ny=8,
            nz=8,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            z_min=0.0,
            z_max=1.0,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )

        # Uniform field: curl(B) = 0 => J = 0 => dB/dt = 0
        B_init = jnp.zeros((8, 8, 8, 3))
        B_init = B_init.at[..., 2].set(B0)

        state = State.zeros(8, 8, 8)
        state = state.replace(B=B_init, n=jnp.ones((8, 8, 8)) * 1e19)

        model = ResistiveMHD(eta=eta)
        rhs = model.compute_rhs(state, geom)

        # RHS should be essentially zero for uniform field
        # (small numerical noise from finite differences is acceptable)
        max_rhs = jnp.max(jnp.abs(rhs.B))
        assert max_rhs < 1e-10, f"RHS for uniform field should be ~0, got {float(max_rhs)}"

    def test_perturbation_shape_preservation(self):
        """Sinusoidal shape should be preserved during decay.

        Resistive diffusion damps amplitude but maintains the sinusoidal
        spatial profile. Check correlation between initial and final profiles.
        """
        eta = 1e-3
        dB = 0.01

        Lz = 1.0
        nz = 32
        geom = Geometry(
            nx=4,
            ny=4,
            nz=nz,
            x_min=0.0,
            x_max=0.1,
            y_min=0.0,
            y_max=0.1,
            z_min=0.0,
            z_max=Lz,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )

        k = 2 * jnp.pi / Lz
        z = geom.z_grid
        B_init = jnp.zeros((4, 4, nz, 3))
        By_init = dB * jnp.sin(k * z)
        B_init = B_init.at[..., 1].set(By_init)

        state = State.zeros(4, 4, nz)
        state = state.replace(B=B_init, n=jnp.ones((4, 4, nz)) * 1e19)

        model = ResistiveMHD(eta=eta)
        dx_min = min(geom.dx, geom.dy, geom.dz)
        diffusivity = eta / MU0
        dt = 0.2 * dx_min**2 / diffusivity
        n_steps = 100

        def body_fn(i, state):
            return rk4_step(state, dt, model, geom)

        state_final = lax.fori_loop(0, n_steps, body_fn, state)

        # Compute correlation between initial and final profiles
        By_final = state_final.B[..., 1]
        # Take 1D slice at center of x-y plane
        By_init_1d = By_init[2, 2, :]
        By_final_1d = By_final[2, 2, :]

        # Normalize and compute correlation
        By_init_norm = By_init_1d - jnp.mean(By_init_1d)
        By_final_norm = By_final_1d - jnp.mean(By_final_1d)
        correlation = jnp.sum(By_init_norm * By_final_norm) / (
            jnp.sqrt(jnp.sum(By_init_norm**2)) * jnp.sqrt(jnp.sum(By_final_norm**2))
        )

        # Should have high correlation (same shape, just damped)
        assert correlation > 0.99, (
            f"Profile correlation {float(correlation):.4f} is too low - "
            "sinusoidal shape not preserved during damping"
        )
