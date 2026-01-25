# Key Physics Concepts

Background on the physics implemented in JAX-FRC.

## Flux Function Formulation

In resistive MHD, we solve for the poloidal flux ψ(r,z) instead of the full magnetic field vector. This reduces the problem to 2D axisymmetric geometry.

The magnetic field is recovered from:
```
B_r = -(1/r) ∂ψ/∂z
B_z = (1/r) ∂ψ/∂r
```

## Chodura Resistivity

Anomalous resistivity model for FRC formation that mimics micro-turbulence effects at the plasma boundary:

```
η = η_0 + η_anom * f(j/j_crit)
```

Where:
- `η_0` is the classical Spitzer resistivity
- `η_anom` is the anomalous enhancement
- `f` is a threshold function that activates at high current density

## Hall Term

Two-fluid effect that separates electron and ion motion:

```
E_Hall = (J × B) / (n e)
```

This term is crucial for capturing kinetic stabilization in FRCs. Without it, MHD codes predict instabilities that don't occur in experiments.

## Semi-Implicit Stepping

Numerical technique to handle stiff Whistler waves in extended MHD without requiring extremely small time steps:

```
(I - Δt² L_Hall) ΔB^{n+1} = Explicit terms
```

The Hall term is treated implicitly, allowing the Alfven CFL condition to determine the timestep rather than the much more restrictive Whistler CFL.

## Delta-f PIC Method

Particle-in-Cell method that simulates deviations from an equilibrium distribution:

```
f(x,v,t) = f_0(x,v) + δf(x,v,t)
```

Particles carry weights `w` representing δf/f. This reduces noise by a factor of 1/δf compared to full-f PIC.

Weight evolution:
```
dw/dt = -(1-w) d ln f_0 / dt
```

## Rigid Rotor Equilibrium

Analytical equilibrium distribution for FRCs:

```
f_0 = n_0 (m/(2πT))^(3/2) exp(-m/(2T)(v_r² + (v_θ - Ωr)² + v_z²))
```

Where:
- `n_0` is the peak density
- `T` is the temperature
- `Ω` is the rotation frequency

This distribution is used as the background for delta-f simulations.

## Physical Constants

```python
MU0 = 1.2566e-6   # Permeability of free space [H/m]
QE = 1.602e-19    # Elementary charge [C]
ME = 9.109e-31    # Electron mass [kg]
MI = 1.673e-27    # Ion mass (proton) [kg]
KB = 1.381e-23    # Boltzmann constant [J/K]
```

## Characteristic Scales

### Alfven Speed
```
v_A = B / √(μ_0 ρ)
```

### Ion Cyclotron Frequency
```
Ω_ci = q B / m_i
```

### Ion Skin Depth
```
d_i = c / ω_pi = √(m_i / (μ_0 n q²))
```

### Plasma Beta
```
β = 2 μ_0 n k_B T / B²
```
