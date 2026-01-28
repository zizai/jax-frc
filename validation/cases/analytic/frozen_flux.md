Here is the formulation of the frozen flux problem for validating Magnetohydrodynamics (MHD) solvers in the high Magnetic Reynolds number regime, along with a discussion of the solutions.

---

### **1. Introduction: The Frozen Flux Approximation**

In Ideal MHD, the electrical resistivity of the fluid is assumed to be zero. This corresponds to the limit where the Magnetic Reynolds number, defined as ( R_m = \frac{L U}{\eta} ), approaches infinity (( R_m \gg 1 )). In this regime, the diffusion of the magnetic field is negligible compared to its advection.

The fundamental consequence of this approximation is **Alfvén's Frozen Flux Theorem**, which states that the magnetic flux through a surface moving with the fluid remains constant. For a numerical solver, correctly capturing this behavior without introducing excessive numerical dissipation (which acts as artificial resistivity) is a critical validation step.

---

### **2. Mathematical Formulation**

The evolution of the magnetic field (\mathbf{B}) is governed by the induction equation.

#### **The Induction Equation**

The general form is:
[
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
]
Where:

* (\mathbf{B}) is the magnetic field.
* (\mathbf{u}) is the fluid velocity field.
* (\eta) is the magnetic diffusivity (inversely proportional to conductivity).

#### **High (R_m) Limit (Ideal MHD)**

When (R_m \gg 1), the resistive term (\eta \nabla^2 \mathbf{B}) is neglected. The equation simplifies to the ideal induction equation:
[
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u} \times \mathbf{B})
]
Using vector identities, this can be expanded to the advective form (assuming (\nabla \cdot \mathbf{B} = 0)):
[
\frac{\partial \mathbf{B}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{B} = (\mathbf{B} \cdot \nabla)\mathbf{u} - \mathbf{B}(\nabla \cdot \mathbf{u})
]
For an incompressible fluid ((\nabla \cdot \mathbf{u} = 0)), this simplifies further to:
[
\frac{D \mathbf{B}}{D t} = (\mathbf{B} \cdot \nabla)\mathbf{u}
]
Where (\frac{D}{Dt}) is the material derivative. This equation implies that the magnetic field lines behave like material lines dyed into the fluid; they are stretched and rotated by gradients in the velocity field but cannot break or reconnect.

---

### **3. Validation Benchmark: Circular Advection of a Magnetic Loop**

To validate a solver's ability to maintain the "frozen-in" condition, we design a 2D problem where a localized magnetic structure is advected by a known velocity field. If the solver is accurate, the magnetic structure should return to its initial state after one full rotation without loss of amplitude (numerical diffusion) or distortion (dispersion).

#### **Problem Setup**

* **Domain:** ( x, y \in [-1, 1] )

* **Grid:** Periodic boundaries or open boundaries sufficiently far from the loop.

* **Velocity Field (Prescribed):** Rigid body rotation.
  [
  u_x(x, y) = -\omega y, \quad u_y(x, y) = \omega x
  ]
  where (\omega) is the angular velocity (e.g., (\omega = 2\pi)).

* **Initial Magnetic Field:**
  To ensure (\nabla \cdot \mathbf{B} = 0), we define the field using a magnetic vector potential (A_z).
  [
  \mathbf{B} = \nabla \times (A_z \hat{z}) = \left( \frac{\partial A_z}{\partial y}, -\frac{\partial A_z}{\partial x}, 0 \right)
  ]
  Let the initial vector potential be a localized Gaussian or compact cylinder centered at ((x_0, 0)):
  [
  A_z(x, y, t=0) = \begin{cases}
  A_0 (1 - r^2/R^2)^2 & \text{if } r < R \
  0 & \text{if } r \ge R
  \end{cases}
  ]
  where (r = \sqrt{(x-x_0)^2 + y^2}) is the distance from the center of the magnetic loop, and (R) is the loop radius (e.g., (R=0.3)).

---

### **4. Exact Solution**

Since the governing equation is purely advective and the velocity field corresponds to a rigid rotation, the exact solution is simply the initial condition rotated about the origin by an angle (\theta = \omega t).

At time (t), the center of the loop moves to:
[
x_c(t) = x_0 \cos(\omega t)
]
[
y_c(t) = x_0 \sin(\omega t)
]
The magnetic potential (A_z) is a scalar invariant in 2D ideal MHD (for (\mathbf{u} \cdot \hat{z} = 0)):
[
A_z(\mathbf{x}, t) = A_z(\mathbf{R}*{-\omega t} \mathbf{x}, 0)
]
where (\mathbf{R}*{-\omega t}) is the rotation matrix for angle (-\omega t).

Consequently, the magnetic field (\mathbf{B}) rotates with the vector potential. The magnitude of the peak magnetic field (|\mathbf{B}|_{max}) should remain constant for all (t).

---

### **5. Discussion of Solutions and Solver Requirements**

In a numerical implementation, the solution will deviate from the exact frozen flux behavior due to discretization errors.

#### **1. Numerical Diffusion (Artificial Resistivity)**

Even if explicit resistivity (\eta) is set to zero, truncation errors in the discretization of the advection term (\nabla \times (\mathbf{u} \times \mathbf{B})) act as an implicit diffusion term.
[
\frac{\partial \mathbf{B}}{\partial t} \approx \nabla \times (\mathbf{u} \times \mathbf{B}) + \eta_{num} \nabla^2 \mathbf{B}
]

* **Benchmark Metric:** Measure the decay of the total magnetic energy (E_m = \frac{1}{2} \int |\mathbf{B}|^2 dV) over time. In the exact frozen flux solution, (E_m) is constant. A rapid decay indicates high numerical dissipation, effectively lowering the solver's "effective" Magnetic Reynolds number.

#### **2. Preservation of Topology ((\nabla \cdot \mathbf{B} = 0))**

The frozen flux constraint implies that field lines cannot reconnect.

* **Benchmark Metric:** Monitor the divergence error (\nabla \cdot \mathbf{B}). While analytically zero, numerical schemes (like cell-centered finite volume without divergence cleaning or constrained transport) can generate non-zero divergence, creating "monopoles" that violate physics and destabilize the solution.

#### **3. Dispersion and Shape Preservation**

High-order schemes might preserve energy better but introduce dispersive errors (oscillations or "ripples" near sharp gradients).

* **Benchmark Metric:** Compare the L2-error norm between the numerical solution (\mathbf{B}*{num}) and the exact analytical solution (\mathbf{B}*{exact}) after one full rotation period (T = 2\pi/\omega):
  [
  \epsilon_{L2} = \sqrt{ \int_{\Omega} |\mathbf{B}*{num}(T) - \mathbf{B}*{exact}(T)|^2 dV }
  ]

### **Conclusion**

For a solver designed for (R_m \gg 1), the "Frozen Flux" benchmark via rigid body rotation is the standard test. A successful solver must transport the magnetic structure while maintaining:

1. **Peak amplitude** (minimal numerical diffusion).
2. **Symmetry** (minimal phase error/dispersion).
3. **Divergence-free constraint** (using Constrained Transport or projection methods).
