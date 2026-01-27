"""State container for 3D plasma simulations."""

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ParticleState:
    """Particle state for hybrid kinetic model."""
    x: Array      # Positions, shape (n_particles, 3)
    v: Array      # Velocities, shape (n_particles, 3)
    w: Array      # Delta-f weights, shape (n_particles,)
    species: str  # Particle species identifier


@dataclass(frozen=True)
class State:
    """State container for 3D plasma simulations.

    All fields are 3D arrays with shape (nx, ny, nz) for scalars
    and (nx, ny, nz, 3) for vectors.
    """
    B: Array           # Magnetic field [T], shape (nx, ny, nz, 3)
    E: Array           # Electric field [V/m], shape (nx, ny, nz, 3)
    n: Array           # Number density [m^-3], shape (nx, ny, nz)
    p: Array           # Pressure [Pa], shape (nx, ny, nz)
    v: Optional[Array] = None  # Velocity [m/s], shape (nx, ny, nz, 3)
    Te: Optional[Array] = None # Electron temp [J], shape (nx, ny, nz)
    Ti: Optional[Array] = None # Ion temp [J], shape (nx, ny, nz)
    particles: Optional[ParticleState] = None

    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int) -> "State":
        """Create zero-initialized state."""
        return cls(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.zeros((nx, ny, nz)),
            p=jnp.zeros((nx, ny, nz)),
        )

    def replace(self, **kwargs) -> "State":
        """Return new State with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register as JAX pytree
def _state_flatten(state):
    children = (state.B, state.E, state.n, state.p, state.v, state.Te, state.Ti, state.particles)
    aux_data = None
    return children, aux_data


def _state_unflatten(aux_data, children):
    B, E, n, p, v, Te, Ti, particles = children
    return State(B=B, E=E, n=n, p=p, v=v, Te=Te, Ti=Ti, particles=particles)


jax.tree_util.register_pytree_node(State, _state_flatten, _state_unflatten)


def _particle_state_flatten(state):
    return (state.x, state.v, state.w), state.species


def _particle_state_unflatten(species, children):
    x, v, w = children
    return ParticleState(x=x, v=v, w=w, species=species)


jax.tree_util.register_pytree_node(ParticleState, _particle_state_flatten, _particle_state_unflatten)
