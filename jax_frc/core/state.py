"""Simulation state containers."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import Array
import jax

@dataclass(frozen=True)
class ParticleState:
    """State container for kinetic particles."""

    x: Array          # Positions (n_particles, 3)
    v: Array          # Velocities (n_particles, 3)
    w: Array          # Delta-f weights (n_particles,)
    species: str      # "ion", "beam", etc.

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

@dataclass(frozen=True)
class State:
    """Complete simulation state at a single time."""

    # Scalar fields (nr, nz)
    psi: Array        # Poloidal flux function
    n: Array          # Number density
    p: Array          # Pressure

    # Vector fields (nr, nz, 3)
    B: Array          # Magnetic field
    E: Array          # Electric field
    v: Array          # Fluid velocity

    # Particles (optional, for hybrid)
    particles: Optional[ParticleState]

    # Metadata
    time: float
    step: int

    @classmethod
    def zeros(cls, nr: int, nz: int, with_particles: bool = False,
              n_particles: int = 0) -> "State":
        """Create a zero-initialized state."""
        particles = None
        if with_particles and n_particles > 0:
            particles = ParticleState(
                x=jnp.zeros((n_particles, 3)),
                v=jnp.zeros((n_particles, 3)),
                w=jnp.zeros(n_particles),
                species="ion"
            )

        return cls(
            psi=jnp.zeros((nr, nz)),
            n=jnp.zeros((nr, nz)),
            p=jnp.zeros((nr, nz)),
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=particles,
            time=0.0,
            step=0
        )

    def replace(self, **kwargs) -> "State":
        """Return new State with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)

# Register State as a JAX pytree for JIT compatibility
def _state_flatten(state):
    children = (state.psi, state.n, state.p, state.B, state.E, state.v,
                state.particles, state.time, state.step)
    aux_data = None
    return children, aux_data

def _state_unflatten(aux_data, children):
    psi, n, p, B, E, v, particles, time, step = children
    return State(psi=psi, n=n, p=p, B=B, E=E, v=v,
                 particles=particles, time=time, step=step)

jax.tree_util.register_pytree_node(State, _state_flatten, _state_unflatten)
