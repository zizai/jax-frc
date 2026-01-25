"""Simulation result types for consistent API across all models.

This module provides dataclasses that standardize the return types from
simulation functions, making it easier to work with results from different
physics models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class SimulationResult:
    """Container for simulation results with consistent API.

    All simulation models should return this type from their run_simulation()
    functions to ensure a consistent interface for post-processing and analysis.

    Attributes:
        model_name: Name of the physics model used (e.g., "resistive_mhd")
        final_time: Simulation end time [s]
        n_steps: Number of timesteps completed
        grid_shape: Shape of the computational grid (nr, nz)
        grid_spacing: Grid spacing (dr, dz) in [m]

        # Field data - varies by model
        psi: Final poloidal flux (resistive MHD)
        B: Final magnetic field components (extended MHD, hybrid)
        velocity: Final velocity field (if evolved)
        density: Final density field (if evolved)
        temperature: Final temperature field (if evolved)
        particles: Final particle state (hybrid kinetic)

        # History data
        history: Time history of key quantities
        diagnostics: Dict of diagnostic quantities
    """

    model_name: str
    final_time: float
    n_steps: int
    grid_shape: Tuple[int, ...]
    grid_spacing: Tuple[float, ...]

    # MHD fields (optional depending on model)
    psi: Optional[Array] = None
    B: Optional[Tuple[Array, ...]] = None
    velocity: Optional[Tuple[Array, ...]] = None
    density: Optional[Array] = None
    temperature: Optional[Array] = None
    pressure: Optional[Array] = None

    # Circuit quantities (resistive MHD)
    I_coil: Optional[float] = None

    # Particle data (hybrid kinetic)
    particles: Optional[Dict[str, Array]] = None

    # History and diagnostics
    history: Optional[Any] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def get_magnetic_energy(self, mu0: float = 1.2566e-6) -> float:
        """Compute total magnetic energy from final state.

        Returns:
            Total magnetic energy [J/m] (integrated over 2D domain)
        """
        if self.B is not None:
            b_x, b_y, b_z = self.B
            B_sq = b_x**2 + b_y**2 + b_z**2
            dr, dz = self.grid_spacing
            return float(jnp.sum(B_sq) * dr * dz / (2 * mu0))
        elif self.psi is not None:
            # For psi formulation, need to compute B from psi
            # This is model-specific, return None for now
            return float('nan')
        return float('nan')

    def get_max_field(self) -> float:
        """Get maximum magnetic field magnitude.

        Returns:
            Maximum |B| in the domain [T]
        """
        if self.B is not None:
            b_x, b_y, b_z = self.B
            B_mag = jnp.sqrt(b_x**2 + b_y**2 + b_z**2)
            return float(jnp.max(B_mag))
        elif self.psi is not None:
            return float(jnp.max(jnp.abs(self.psi)))
        return float('nan')

    def summary(self) -> str:
        """Return a human-readable summary of the simulation result.

        Returns:
            Multi-line string with key statistics
        """
        lines = [
            f"Simulation Result: {self.model_name}",
            f"  Grid: {self.grid_shape}, spacing: {self.grid_spacing}",
            f"  Steps: {self.n_steps}, final time: {self.final_time:.6e} s",
        ]

        if self.B is not None:
            lines.append(f"  Max |B|: {self.get_max_field():.6e} T")
            lines.append(f"  Magnetic energy: {self.get_magnetic_energy():.6e} J/m")

        if self.psi is not None:
            lines.append(f"  Max |psi|: {float(jnp.max(jnp.abs(self.psi))):.6e}")

        if self.I_coil is not None:
            lines.append(f"  Final I_coil: {self.I_coil:.6e} A")

        if self.particles is not None:
            n_particles = self.particles.get('x', jnp.array([])).shape[0]
            lines.append(f"  Particles: {n_particles}")

        return "\n".join(lines)


@dataclass
class ParticleState:
    """Container for particle data in hybrid kinetic simulations.

    Attributes:
        x: Particle positions, shape (n_particles, 3)
        v: Particle velocities, shape (n_particles, 3)
        w: Particle weights for delta-f method, shape (n_particles,)
    """

    x: Array
    v: Array
    w: Array

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.x.shape[0]

    def to_dict(self) -> Dict[str, Array]:
        """Convert to dictionary format."""
        return {"x": self.x, "v": self.v, "w": self.w}

    @classmethod
    def from_dict(cls, data: Dict[str, Array]) -> "ParticleState":
        """Create from dictionary format."""
        return cls(x=data["x"], v=data["v"], w=data["w"])
