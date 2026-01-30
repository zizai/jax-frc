## Numerical schemes for MHD

Here is a discussion of the numerical schemes tailored for the three regimes of MHD: **Ideal**, **Resistive**, and **Hall MHD**. Each regime presents distinct mathematical properties (Hyperbolic, Parabolic, and Dispersive), requiring specific numerical strategies.

---

### **1. Ideal MHD (The Hyperbolic Regime)**

**Physics:** The fluid is a perfect conductor ($R_m \to \infty$). The system is governed by conservation laws and supports discontinuous solutions (shocks).
**Mathematical Nature:** Hyperbolic system of Partial Differential Equations (PDEs).

#### **A. Spatial Discretization: Godunov-Type Finite Volume**

The standard for modern Ideal MHD codes (like Athena++, FLASH, PLUTO) is the **Finite Volume Method (FVM)** using a Godunov approach.

* **Concept:** Variables are averaged over a cell. Fluxes at cell interfaces are computed by solving a "Riemann Problem" (the evolution of a discontinuity between two states).
* **Riemann Solvers:**

  * **Roe Solver:** Accurate but expensive; requires expanding the full characteristic eigenstructure of MHD (7 waves).
  * **HLLD (Harten-Lax-van Leer-Discontinuities):** The "Gold Standard" for Ideal MHD. It approximates the Riemann fan with 5 waves (Fast, Alfvén, Contact discontinuities). It is robust, efficient, and resolves Alfvén waves exactly.
  * **HLLE:** More diffusive, uses only 2 waves. Good for stability in extreme environments (e.g., strong shocks) but loses detail.

#### **B. Reconstruction (High-Order Accuracy)**

To achieve high accuracy, the values at the cell interfaces are reconstructed from cell centers using interpolation.

* **PPM (Piecewise Parabolic Method):** Excellent for sharp discontinuities.
* **WENO (Weighted Essentially Non-Oscillatory):** High order (5th+), prevents spurious oscillations near shocks by weighting smoothness.

#### **C. Divergence Handling**

In Ideal MHD, preserving $\nabla \cdot \mathbf{B} = 0$ is critical to prevent the generation of unphysical forces.

* **Constrained Transport (CT):** The magnetic field is treated as an area-weighted flux on cell **faces** (staggered grid). Fluxes are updated via edge-centered electric fields ($\mathbf{E}$). This guarantees $\nabla \cdot \mathbf{B} = 0$ to machine precision.

---

### **2. Resistive MHD (The Parabolic Regime)**

**Physics:** Magnetic field diffuses due to finite resistivity ($\eta \neq 0$). Shocks are smoothed out.
**Mathematical Nature:** Mixed Hyperbolic-Parabolic system. The diffusion term is "stiff."

#### **A. Operator Splitting**

A common approach is to split the equation into an Ideal (Advection) part and a Resistive (Diffusion) part:
[ \frac{\partial \mathbf{B}}{\partial t} = \underbrace{\nabla \times (\mathbf{u} \times \mathbf{B})}*{\text{Hyperbolic}} + \underbrace{\eta \nabla^2 \mathbf{B}}*{\text{Parabolic}} ]

* **Step 1:** Advance the Ideal part using the explicit Godunov schemes described above.
* **Step 2:** Advance the Resistive part using a different scheme suitable for diffusion.

#### **B. Explicit Diffusion (with Super Time-Stepping)**

If you use a simple explicit scheme for diffusion, the time step is limited by $\Delta t \sim \Delta x^2 / \eta$. For high resolution, this is prohibitively small.

* **Scheme:** **Super Time-Stepping (STS)** (Runge-Kutta Legendre).
* **Concept:** Instead of taking many tiny stable steps, STS takes a series of *unstable* steps (like a "leap of faith") that average out to a stable solution at the end of a "super step." This accelerates the calculation effectively.

#### **C. Implicit Methods (Crank-Nicolson)**

For low $R_m$ (high resistivity), the diffusion timescale is much faster than the fluid flow.

* **Scheme:** Implicit-Explicit (IMEX) or purely Implicit (Backward Euler, Crank-Nicolson).
* **Concept:** Solves a large linear system matrix inversion ($Ax=b$) at every time step.
* **Pros:** Unconditionally stable; you can take large time steps limited only by the flow advection ($\Delta t \sim \Delta x/u$).
* **Cons:** Hard to parallelize; computationally expensive per step.

---

### **3. Hall MHD (The Dispersive Regime)**

**Physics:** Electrons and ions decouple at small scales ($d_i$). Supports **Whistler Waves**.
**Mathematical Nature:** Dispersive Hyperbolic. The highest wave speed depends on the grid size ($v \propto k \sim 1/\Delta x$).

#### **A. The "Whistler Catastrophe"**

In Hall MHD, the wave speed scales as $V_{whistler} \propto B / \Delta x$.
As you refine the grid ($\Delta x \to 0$), the wave speed goes to infinity.

* **Explicit Constraint:** $\Delta t \propto \Delta x^2$.
  Even though the Hall term looks like a flux (hyperbolic), numerically it behaves like diffusion (quadratic scaling) because the wave speed grows as the grid shrinks.

#### **B. Hall-MHD Riemann Solvers**

Some modern codes (like implementations in Athena++) extend HLLD to include the Hall term.

* **HLLD-H:** Modification of the HLLD solver to account for the dispersive characteristics.
* **Limitation:** Strictly explicit. Only efficient if the Hall scale is barely resolved. If you resolve the Hall scale deeply, the time step vanishes.

#### **C. Semi-Implicit & Implicit Methods**

Because of the quadratic time-step constraint, implicit methods are almost mandatory for serious Hall MHD simulations.

* **Hall-Implicit:** The Ideal MHD part is treated explicitly, but the Hall term ($\mathbf{J} \times \mathbf{B}$) is solved implicitly.
* **Hyper-Resistivity:** Sometimes, a small amount of "artificial hyper-resistivity" ($\nabla^4 \mathbf{B}$) is added to damp the highest frequency whistler waves, stabilizing the code without fully implicit solving (at the cost of accuracy).

---

### **Summary Comparison Table**

| Regime        | Dominant Physics       | Numerical Challenge                              | Preferred Scheme Strategy                                                       |
| :------------ | :--------------------- | :----------------------------------------------- | :------------------------------------------------------------------------------ |
| **Ideal**     | Advection, Shocks      | Discontinuities                                  | **Explicit FVM (Godunov)** with **HLLD** + **CT**.                              |
| **Resistive** | Diffusion              | Stiff Time Step ($\Delta t \sim \Delta x^2$)     | **Operator Splitting**: Explicit Advection + **STS** or **Implicit** Diffusion. |
| **Hall**      | Dispersion (Whistlers) | Quadratic Stiffness ($\Delta t \sim \Delta x^2$) | **Semi-Implicit** (Implicit Hall term) or specialized **Hall-Riemann Solvers**. |
