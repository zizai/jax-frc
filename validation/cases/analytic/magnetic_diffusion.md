Here is the formulation of the magnetic diffusion problem for validating Magnetohydrodynamics (MHD) solvers in the low Magnetic Reynolds number regime.

---

### **1. Introduction: The Diffusion Approximation**

In the resistive limit of MHD, the electrical resistivity (\eta) of the fluid plays a dominant role. This corresponds to the limit where the Magnetic Reynolds number, defined as ( R_m = \frac{L U}{\eta} ), approaches zero (( R_m \ll 1 )). In this regime, the advection of the magnetic field by the fluid flow is negligible compared to the resistive dissipation.

This approximation describes the behavior of magnetic fields in highly resistive media, such as liquid metals, or at very small length scales in plasmas. For a numerical solver, correctly capturing this behavior validates the implementation of the parabolic diffusion terms and the stability of the time-integration scheme.

---

### **2. Mathematical Formulation**

The evolution of the magnetic field (\mathbf{B}) is governed by the induction equation.

#### **The Induction Equation**

Recall the general form:
[
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}
]

#### **Low (R_m) Limit (Resistive MHD)**

When (R_m \ll 1), or in the case of a static fluid ((\mathbf{u} = \mathbf{0})), the advective term (\nabla \times (\mathbf{u} \times \mathbf{B})) vanishes or becomes negligible. The equation simplifies to the vector diffusion equation (heat equation for magnetism):
[
\frac{\partial \mathbf{B}}{\partial t} = \eta \nabla^2 \mathbf{B}
]
Unlike the hyperbolic wave-like nature of the frozen flux problem, this is a parabolic partial differential equation. It implies that any initial structure in the magnetic field will smooth out and decay over time, ultimately reaching a uniform zero state (in the absence of external drivers).

---

### **3. Validation Benchmark: Decay of a Static Magnetic Mode**

To validate the diffusive part of a solver, we simulate the decay of a specific Fourier mode of the magnetic field in a static domain. This tests the accuracy of the spatial discretization of the Laplacian operator and the stability of the time-stepping method.

#### **Problem Setup**

* **Domain:** A 2D square domain ( x, y \in [0, L] ) (e.g., ( L = 2\pi )).

* **Grid:** Periodic boundary conditions in both (x) and (y).

* **Velocity Field:** Static fluid.
  [
  \mathbf{u} = \mathbf{0}
  ]

* **Parameters:** Constant magnetic diffusivity (\eta) (e.g., (\eta = 0.01)).

* **Initial Magnetic Field:**
  To satisfy (\nabla \cdot \mathbf{B} = 0), we again define the field using a scalar potential (A_z). We choose a simple harmonic function:
  [
  A_z(x, y, t=0) = A_0 \sin(k_x x) \sin(k_y y)
  ]
  Where (k_x) and (k_y) are wavenumbers. For (L=2\pi), we can choose (k_x = 1, k_y = 1).

  The corresponding magnetic field is:
  [
  \mathbf{B} = \left( \frac{\partial A_z}{\partial y}, -\frac{\partial A_z}{\partial x}, 0 \right) = A_0 \left( k_y \sin(k_x x)\cos(k_y y), -k_x \cos(k_x x)\sin(k_y y), 0 \right)
  ]

---

### **4. Exact Solution**

For the diffusion equation (\frac{\partial A_z}{\partial t} = \eta \nabla^2 A_z), we substitute the trial solution (A_z(x, y, t) = T(t) \sin(k_x x) \sin(k_y y)).

The Laplacian of the spatial part is:
[
\nabla^2 (\sin(k_x x) \sin(k_y y)) = -(k_x^2 + k_y^2) \sin(k_x x) \sin(k_y y)
]
Let (k^2 = k_x^2 + k_y^2). The time evolution is governed by:
[
\frac{dT}{dt} = -\eta k^2 T \implies T(t) = T(0) e^{-\eta k^2 t}
]

Thus, the exact solution for the vector potential is:
[
A_z(x, y, t) = A_0 e^{-\eta (k_x^2 + k_y^2) t} \sin(k_x x) \sin(k_y y)
]
And the magnetic field components decay at the same rate:
[
\mathbf{B}(x, y, t) = \mathbf{B}(x, y, 0) e^{-\frac{t}{\tau_{decay}}}
]
where the decay time scale is (\tau_{decay} = \frac{1}{\eta (k_x^2 + k_y^2)}).

---

### **5. Discussion of Solutions and Solver Requirements**

Validating the resistive term presents different challenges than the advective term, focusing on stability constraints and the accuracy of gradients.

#### **1. Decay Rate Accuracy**

The primary metric is the error in the decay rate.

* **Benchmark Metric:** Calculate the total magnetic energy (E_m(t) = \frac{1}{2} \int |\mathbf{B}|^2 dV). Analytically, (E_m(t) = E_m(0) e^{-2\eta k^2 t}).
* **Analysis:** The numerical solver will produce a discrete decay rate. Comparing the slope of (\ln(E_m(t))) versus time against the theoretical value (-2\eta k^2) reveals the accuracy of the spatial discretization (second-order central difference, fourth-order, etc.).

#### **2. Time-Stepping Stability (Stiffness)**

The diffusion equation is "stiff."

* **Explicit Schemes:** If the solver uses explicit time stepping (e.g., Runge-Kutta), the time step (\Delta t) is strictly limited by the grid spacing (\Delta x):
  [
  \Delta t < \frac{\Delta x^2}{C \cdot \eta}
  ]
  where (C) depends on the dimension and scheme. As the grid resolution doubles, the required time step decreases by a factor of 4. A solver failing this benchmark will blow up (instability).
* **Implicit Schemes:** Implicit solvers (e.g., Crank-Nicolson) are unconditionally stable for diffusion but involve solving large linear systems. This benchmark validates that the matrix inversion is correct and that large time steps do not degrade temporal accuracy excessively.

#### **3. Divergence Constraint in Resistive Terms**

While the diffusion operator (\nabla^2) naturally commutes with the divergence operator (preserving (\nabla \cdot \mathbf{B} = 0) if it is initially zero), numerical implementations can introduce errors.

* **Validation:** For curvilinear coordinates or unstructured grids, the vector Laplacian definition (\nabla^2 \mathbf{B} = \nabla(\nabla \cdot \mathbf{B}) - \nabla \times (\nabla \times \mathbf{B})) is often used. If the solver incorrectly discretizes these terms, it may generate divergence errors even in a pure diffusion problem.

### **Conclusion**

For a solver operating in low (R_m) regimes or using explicit resistivity for stabilization, the "Static Decay" benchmark is essential. A successful solver must:

1. **Match the theoretical decay rate** (verifying the magnitude of (\eta)).
2. **Maintain stability** according to the appropriate CFL condition (explicit) or solver tolerance (implicit).
3. **Preserve the modal shape** (pure decay without introducing phase shifts or distortions).
