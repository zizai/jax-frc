# Diagnostics

Measurement probes and output utilities for simulation analysis.

## Package Structure

```
jax_frc/diagnostics/
├── probes.py     # Diagnostic probes (Flux, Energy, Beta, Current)
├── merging.py    # Merging-specific diagnostics
└── output.py     # HDF5 checkpoints and time history I/O
```

## DiagnosticSet

Create and manage a collection of diagnostic probes:

```python
from jax_frc.diagnostics import DiagnosticSet, save_time_history, save_checkpoint

# Create default diagnostic set
diagnostics = DiagnosticSet.default_set()

# Measure during simulation
for i in range(steps):
    sim.step()
    results = diagnostics.measure_all(sim.state, geometry)

# Save time history and checkpoint
save_time_history(diagnostics.get_history(), "output/history.csv")
save_checkpoint(sim.state, geometry, "output/checkpoint.h5")
```

## Available Probes

| Probe | Description |
|-------|-------------|
| FluxProbe | Magnetic flux measurements |
| EnergyProbe | Total energy (magnetic + kinetic + thermal) |
| BetaProbe | Plasma beta (pressure ratio) |
| CurrentProbe | Plasma current |

## Merging Diagnostics

Specialized diagnostics for FRC merging simulations:

```python
from jax_frc.diagnostics.merging import MergingDiagnostics

diag = MergingDiagnostics()
metrics = diag.compute(state, geometry)

print(f"Separation: {metrics['separation_dz']}")
print(f"Reconnection rate: {metrics['reconnection_rate']}")
```

### Merging Metrics

| Metric | Description |
|--------|-------------|
| `separation_dz` | Distance between O-points |
| `reconnection_rate` | Flux annihilation rate |
| `merged` | Boolean: FRCs fully merged |

## Output Formats

### Time History (CSV)

```python
save_time_history(history, "output/history.csv")
```

### Checkpoint (HDF5)

```python
save_checkpoint(state, geometry, "output/checkpoint.h5")
```
