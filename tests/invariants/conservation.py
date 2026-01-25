"""Conservation invariants - physical quantities that should be preserved."""
import jax.numpy as jnp
from tests.invariants import Invariant, InvariantResult

class EnergyConservation(Invariant):
    """Total energy should be conserved within tolerance."""

    def __init__(self, rtol: float = 0.01, energy_fn=None):
        self.rtol = rtol
        self.energy_fn = energy_fn or (lambda s: float(jnp.sum(s**2)))

    @property
    def name(self) -> str:
        return "EnergyConservation"

    def check(self, state_before, state_after) -> InvariantResult:
        E_before = self.energy_fn(state_before)
        E_after = self.energy_fn(state_after)

        if abs(E_before) < 1e-15:
            rel_change = 0.0 if abs(E_after) < 1e-15 else float('inf')
        else:
            rel_change = abs(E_after - E_before) / abs(E_before)

        passed = rel_change <= self.rtol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=rel_change,
            tolerance=self.rtol,
            message=f"dE/E = {rel_change:.2e}, E_before={E_before:.2e}, E_after={E_after:.2e}"
        )

class ParticleCountConservation(Invariant):
    """Number of particles should remain constant."""

    @property
    def name(self) -> str:
        return "ParticleCountConservation"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state to be particle position array
        n_before = state_before.shape[0] if hasattr(state_before, 'shape') else len(state_before)
        n_after = state_after.shape[0] if hasattr(state_after, 'shape') else len(state_after)

        passed = n_before == n_after

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=float(abs(n_after - n_before)),
            tolerance=0.0,
            message=f"Particles: {n_before} -> {n_after}"
        )

class MomentumConservation(Invariant):
    """Total momentum should be conserved within tolerance."""

    def __init__(self, rtol: float = 0.01, momentum_fn=None):
        self.rtol = rtol
        self.momentum_fn = momentum_fn or (lambda v: jnp.sum(v, axis=0))

    @property
    def name(self) -> str:
        return "MomentumConservation"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state to be velocity array
        p_before = self.momentum_fn(state_before)
        p_after = self.momentum_fn(state_after)

        p_mag_before = float(jnp.linalg.norm(p_before))
        p_mag_after = float(jnp.linalg.norm(p_after))

        if p_mag_before < 1e-15:
            rel_change = 0.0 if p_mag_after < 1e-15 else float('inf')
        else:
            rel_change = float(jnp.linalg.norm(p_after - p_before)) / p_mag_before

        passed = rel_change <= self.rtol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=rel_change,
            tolerance=self.rtol,
            message=f"dp/p = {rel_change:.2e}"
        )

class FluxConservation(Invariant):
    """Total magnetic flux should be conserved within tolerance."""

    def __init__(self, rtol: float = 0.01):
        self.rtol = rtol

    @property
    def name(self) -> str:
        return "FluxConservation"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state to be psi (flux) array
        flux_before = float(jnp.sum(state_before))
        flux_after = float(jnp.sum(state_after))

        if abs(flux_before) < 1e-15:
            rel_change = 0.0 if abs(flux_after) < 1e-15 else float('inf')
        else:
            rel_change = abs(flux_after - flux_before) / abs(flux_before)

        passed = rel_change <= self.rtol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=rel_change,
            tolerance=self.rtol,
            message=f"dFlux/Flux = {rel_change:.2e}"
        )
