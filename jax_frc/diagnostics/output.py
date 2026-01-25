"""Output and checkpoint utilities for JAX-FRC simulation."""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import json
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State, ParticleState
from jax_frc.core.geometry import Geometry


def save_checkpoint(state: State, geometry: Geometry, path: Union[str, Path],
                    metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save simulation state to HDF5 checkpoint file.

    Args:
        state: Current simulation state
        geometry: Computational geometry
        path: Output file path (.h5 or .hdf5)
        metadata: Optional metadata dictionary
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 checkpoints. Install with: pip install h5py")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        # Save geometry
        geom_grp = f.create_group('geometry')
        geom_grp.attrs['coord_system'] = geometry.coord_system
        geom_grp.attrs['nr'] = geometry.nr
        geom_grp.attrs['nz'] = geometry.nz
        geom_grp.attrs['r_min'] = geometry.r_min
        geom_grp.attrs['r_max'] = geometry.r_max
        geom_grp.attrs['z_min'] = geometry.z_min
        geom_grp.attrs['z_max'] = geometry.z_max

        # Save state fields
        state_grp = f.create_group('state')
        state_grp.create_dataset('psi', data=jnp.asarray(state.psi))
        state_grp.create_dataset('n', data=jnp.asarray(state.n))
        state_grp.create_dataset('p', data=jnp.asarray(state.p))
        state_grp.create_dataset('B', data=jnp.asarray(state.B))
        state_grp.create_dataset('E', data=jnp.asarray(state.E))
        state_grp.create_dataset('v', data=jnp.asarray(state.v))
        state_grp.attrs['time'] = float(state.time)
        state_grp.attrs['step'] = int(state.step)

        # Save particles if present
        if state.particles is not None:
            part_grp = f.create_group('particles')
            part_grp.create_dataset('x', data=jnp.asarray(state.particles.x))
            part_grp.create_dataset('v', data=jnp.asarray(state.particles.v))
            part_grp.create_dataset('w', data=jnp.asarray(state.particles.w))
            part_grp.attrs['species'] = state.particles.species

        # Save metadata
        if metadata:
            meta_grp = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    meta_grp.attrs[key] = value
                else:
                    meta_grp.attrs[key] = json.dumps(value)


def load_checkpoint(path: Union[str, Path]) -> tuple:
    """Load simulation state from HDF5 checkpoint file.

    Args:
        path: Input file path

    Returns:
        Tuple of (state, geometry, metadata)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 checkpoints. Install with: pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with h5py.File(path, 'r') as f:
        # Load geometry
        geom_grp = f['geometry']
        geometry = Geometry(
            coord_system=geom_grp.attrs['coord_system'],
            nr=int(geom_grp.attrs['nr']),
            nz=int(geom_grp.attrs['nz']),
            r_min=float(geom_grp.attrs['r_min']),
            r_max=float(geom_grp.attrs['r_max']),
            z_min=float(geom_grp.attrs['z_min']),
            z_max=float(geom_grp.attrs['z_max']),
        )

        # Load state fields
        state_grp = f['state']
        particles = None

        # Load particles if present
        if 'particles' in f:
            part_grp = f['particles']
            particles = ParticleState(
                x=jnp.array(part_grp['x'][:]),
                v=jnp.array(part_grp['v'][:]),
                w=jnp.array(part_grp['w'][:]),
                species=part_grp.attrs['species']
            )

        state = State(
            psi=jnp.array(state_grp['psi'][:]),
            n=jnp.array(state_grp['n'][:]),
            p=jnp.array(state_grp['p'][:]),
            B=jnp.array(state_grp['B'][:]),
            E=jnp.array(state_grp['E'][:]),
            v=jnp.array(state_grp['v'][:]),
            particles=particles,
            time=float(state_grp.attrs['time']),
            step=int(state_grp.attrs['step'])
        )

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            meta_grp = f['metadata']
            for key in meta_grp.attrs:
                value = meta_grp.attrs[key]
                if isinstance(value, str) and value.startswith('{'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                metadata[key] = value

    return state, geometry, metadata


def save_time_history(history: Dict[str, Any], path: Union[str, Path],
                      format: str = "csv") -> None:
    """Save diagnostic time history to file.

    Args:
        history: Dictionary with 'time' and diagnostic values
        path: Output file path
        format: Output format ("csv" or "json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        # Write CSV
        keys = list(history.keys())
        n_rows = len(history.get('time', []))

        with open(path, 'w') as f:
            # Header
            f.write(','.join(keys) + '\n')
            # Data rows
            for i in range(n_rows):
                row = [str(history[k][i]) if i < len(history[k]) else ''
                       for k in keys]
                f.write(','.join(row) + '\n')

    elif format == "json":
        # Convert arrays to lists for JSON serialization
        json_history = {}
        for key, value in history.items():
            if hasattr(value, 'tolist'):
                json_history[key] = value.tolist()
            elif isinstance(value, list):
                json_history[key] = [float(v) if hasattr(v, 'item') else v
                                     for v in value]
            else:
                json_history[key] = value

        with open(path, 'w') as f:
            json.dump(json_history, f, indent=2)

    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'.")


def load_time_history(path: Union[str, Path]) -> Dict[str, Any]:
    """Load diagnostic time history from file.

    Args:
        path: Input file path (csv or json)

    Returns:
        Dictionary with time history data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)

    elif path.suffix == '.csv':
        history = {}
        with open(path, 'r') as f:
            # Read header
            header = f.readline().strip().split(',')
            for key in header:
                history[key] = []

            # Read data
            for line in f:
                values = line.strip().split(',')
                for key, val in zip(header, values):
                    if val:
                        try:
                            history[key].append(float(val))
                        except ValueError:
                            history[key].append(val)

        return history

    else:
        raise ValueError(f"Unknown file format: {path.suffix}")
