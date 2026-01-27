# Key Physics Concepts

Background on the physics implemented in JAX-FRC.

## Flux Function Formulation

In resistive MHD, we solve for the poloidal flux $\psi(r,z)$ instead of the full magnetic field vector. This reduces the problem to 2D axisymmetric geometry.

The magnetic field is recovered from:

$$
B_r = -\frac{1}{r} \frac{\partial \psi}{\partial z}, \quad
B_z = \frac{1}{r} \frac{\partial \psi}{\partial r}
$$

## Chodura Resistivity

Anomalous resistivity model for FRC formation that mimics micro-turbulence effects at the plasma boundary:

$$
\eta = \eta_0 + \eta_{\rm anom} \cdot f(j/j_{\rm crit})
$$

Where:
- $\eta_0$ is the classical Spitzer resistivity
- $\eta_{\rm anom}$ is the anomalous enhancement
- $f$ is a threshold function that activates at high current density

## Hall Term

Two-fluid effect that separates electron and ion motion:

$$
\mathbf{E}_{\rm Hall} = \frac{\mathbf{J} \times \mathbf{B}}{n e}
$$

This term is crucial for capturing kinetic stabilization in FRCs. Without it, MHD codes predict instabilities that don't occur in experiments.

## Semi-Implicit Stepping

Numerical technique to handle stiff Whistler waves in extended MHD without requiring extremely small time steps:

$$
(I - \Delta t^2 L_{\rm Hall}) \Delta \mathbf{B}^{n+1} = \text{Explicit terms}
$$

The Hall term is treated implicitly, allowing the Alfven CFL condition to determine the timestep rather than the much more restrictive Whistler CFL.

## Delta-f PIC Method

Particle-in-Cell method that simulates deviations from an equilibrium distribution:

$$
f(x,v,t) = f_0(x,v) + \delta f(x,v,t)
$$

Particles carry weights $w$ representing $\delta f/f$. This reduces noise by a factor of $1/\delta f$ compared to full-f PIC.

Weight evolution:

$$
\frac{dw}{dt} = -(1-w) \frac{d \ln f_0}{dt}
$$

## Rigid Rotor Equilibrium

Analytical equilibrium distribution for FRCs:

$$
f_0 = n_0 \left(\frac{m}{2\pi T}\right)^{3/2} \exp\left(-\frac{m}{2T}(v_r^2 + (v_\theta - \Omega r)^2 + v_z^2)\right)
$$

Where:
- $n_0$ is the peak density
- $T$ is the temperature
- $\Omega$ is the rotation frequency

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

$$
v_A = \frac{B}{\sqrt{\mu_0 \rho}}
$$

### Ion Cyclotron Frequency

$$
\Omega_{ci} = \frac{q B}{m_i}
$$

### Ion Skin Depth

$$
d_i = \frac{c}{\omega_{pi}} = \sqrt{\frac{m_i}{\mu_0 n q^2}}
$$

### Plasma Beta

$$
\beta = \frac{2 \mu_0 n k_B T}{B^2}
$$
