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


## FAQ

---

### **1. What does "Higher-Order" mean?**

In numerical analysis, "Order" refers to the **Order of Accuracy**. It describes how quickly the error in your solution decreases as you make the grid cells ($\Delta x$) smaller.

Mathematically, it comes from the Taylor Series expansion. If you approximate a function $f(x)$, the error $E$ scales with the grid size:
[ E \propto (\Delta x)^n ]
Here, **$n$** is the order.

* **1st Order (Low Accuracy):**

  * **Concept:** You assume the fluid inside a cell is a flat, constant block (Piecewise Constant).
  * **Behavior:** It is very stable but extremely "diffusive." Sharp features (like a peak) will smear out and flatten rapidly, as if you added thick syrup to the simulation.
  * **Error:** If you halve the grid size, the error drops by factor of **2** ($2^1$).

* **2nd Order (Standard):**

  * **Concept:** You assume the fluid inside a cell has a slope (Piecewise Linear). You reconstruct the value at the interface by drawing a straight line from the cell center.
  * **Behavior:** The standard for most codes. It captures gradients well but can create false "ringing" (oscillations) near shocks.
  * **Error:** If you halve the grid size, the error drops by factor of **4** ($2^2$).

* **Higher-Order (3rd, 4th, 5th...):**

  * **Concept:** You fit a parabola or a complex polynomial curve inside the cells (Piecewise Parabolic/Polynomial). Methods like **PPM** (Piecewise Parabolic Method) or **WENO** (Weighted Essentially Non-Oscillatory) are high-order.
  * **Behavior:** Extremely accurate. It can transport a complex shape across the grid for a long time without distorting it.
  * **Cost:** Much more computationally expensive per step, but you need fewer grid cells to get the same answer.

---

### **2. Why does the wave speed grow as the grid shrinks? (Hall MHD)**

This phenomenon is unique to **Dispersive Waves** (like Whistler waves in Hall MHD or deep water waves).

In standard fluid dynamics (sound waves), the wave frequency $\omega$ is linearly proportional to the wavenumber $k$ (where $k \propto 1/\text{wavelength}$).
[ \omega = c_s \cdot k ]
The wave speed is $v = \frac{\omega}{k} = c_s$. This is **constant**. No matter how small the wave is, it travels at the speed of sound.

**In Hall MHD (Whistlers):**
The physics dictate that the frequency scales with the **square** of the wavenumber:
[ \omega \propto B \cdot k^2 ]

Therefore, the wave speed $v$ depends on $k$:
[ v = \frac{\omega}{k} \propto B \cdot k ]

**The "Grid" Connection:**
Your computational grid ($\Delta x$) determines the smallest wave you can support. The maximum wavenumber is:
[ k_{max} \approx \frac{\pi}{\Delta x} ]

Substitute this into the speed equation:
[ v_{max} \propto B \cdot \frac{1}{\Delta x} ]

**The Consequence:**

* If you refine your grid by 2x (make $\Delta x$ half as big), the physical whistle waves that fit in that tiny grid travel **2x faster**.
* Since the time step limit (CFL) is $\Delta t < \Delta x / v$, and $v$ grows as $\Delta x$ shrinks, your time step drops by **4x** ($\Delta x^2$). This is why Hall MHD is computationally brutal.

---

### **3. Why use "Many Waves" in a Riemann Solver?**

When a numerical solver calculates the flux between two cells, it solves a "Riemann Problem": *What happens when two different fluid states touch?*

The solution is a "fan" of waves radiating outward.

#### **Ideal MHD has 7 Physical Waves:**

1. **Fast Shock/Wave (Left)**
2. **Alfvén Wave (Left)**
3. **Slow Shock/Wave (Left)**
4. **Contact Discontinuity (Entropy)** (The middle interface)
5. **Slow Shock/Wave (Right)**
6. **Alfvén Wave (Right)**
7. **Fast Shock/Wave (Right)**

#### **The Solvers:**

* **HLLE (2-Wave Solver):**

  * **Assumption:** It only models the fastest wave to the left (#1) and the fastest to the right (#7).
  * **Result:** It ignores everything in the middle (#2, #3, #4, #5, #6). It averages them into a single diffusive blur.
  * **Problem:** If you are studying **Alfvénic turbulence** (which depends on waves #2 and #6), HLLE will delete your physics. The waves will decay instantly due to numerical error.

* **HLLD (5-Wave Solver):**

  * **Assumption:** It explicitly models the Fast waves (#1, #7), the Alfvén waves (#2, #6), and the Contact Discontinuity (#4).
  * **Result:** It captures the rotation of the magnetic field (Alfvén waves) and density changes (Contact) sharply.
  * **Benefit:** It is much less diffusive. It allows magnetic structures to propagate for a long time without fading away.

**Summary:** You use "many waves" (HLLD or Roe) because you want to capture the **internal structure** of the flow. Using "few waves" (HLLE) is safer (more stable) but acts like blurring your simulation with a low-quality filter.

### Why is Hall MHD a lot slower than Ideal MHD?

The reason Hall MHD runs significantly slower than Ideal MHD comes down to a fundamental difference in the mathematical structure of the equations, which creates a "pick your poison" scenario for numerical solvers: either you must take **exponentially smaller time steps**, or you must perform **extremely expensive calculations** at each step.

Here is the breakdown of why this happens.

#### 1. The "Whistler Catastrophe" (Explicit Solvers)

If you use a standard explicit solver (like the ones used for Ideal MHD), the simulation speed is dictated by the **CFL Condition**: the time step $\Delta t$ must be small enough that information doesn't cross more than one grid cell per step.

* **Ideal MHD (Hyperbolic):**
  The fastest wave is the Fast Magnetosonic wave. Its speed ($V_{fast}$) is roughly constant regardless of your grid resolution.
  [ \Delta t_{Ideal} \approx \frac{\Delta x}{V_{fast}} ]
  If you refine your grid by 2x, your time step drops by **2x**. This is a **linear** cost.

* **Hall MHD (Dispersive):**
  The fastest wave is the **Whistler wave**. As discussed previously, its speed depends on the grid size: $V_{whistler} \propto B / \Delta x$. The smaller your cells, the faster the waves travel.
  [ \Delta t_{Hall} \approx \frac{\Delta x}{V_{whistler}} \approx \frac{\Delta x}{(1/\Delta x)} \approx (\Delta x)^2 ]
  If you refine your grid by 2x, your time step drops by **4x**.

  **The result:** To reach the same physical time (e.g., $t=1.0$), a Hall MHD simulation might need **thousands or millions more time steps** than an Ideal MHD simulation, purely to keep the solver from crashing.

#### 2. The Cost of Implicit Solvers

To avoid the tiny time steps mentioned above, researchers often switch to **Implicit Solvers** (e.g., Crank-Nicolson). These allow you to take large time steps ($\Delta t_{Hall} \approx \Delta t_{Ideal}$).

However, there is a catch:

* **Ideal MHD (Explicit):**
  Update rule: `New_State = Old_State + Fluxes`.
  This is just simple multiplication and addition. It is extremely fast (microseconds per cell).
* **Hall MHD (Implicit):**
  Update rule: `Solve Matrix_A * New_State = Old_State`.
  You must solve a massive system of linear equations at *every single time step*.

  * This requires **inverting a matrix** or using iterative solvers (like GMRES or Conjugate Gradient).
  * It requires global communication across the entire computer cluster (bad for parallelization).
  * **The result:** A single time step in Implicit Hall MHD can take **10x to 100x more CPU time** than a single step in Ideal MHD.

#### 3. The "Stiffness" of the Hall Term

Mathematically, the Hall term $\nabla \times (\frac{\mathbf{J} \times \mathbf{B}}{ne})$ involves the current density $\mathbf{J}$, which itself is a derivative of $\mathbf{B}$ ($\mathbf{J} \propto \nabla \times \mathbf{B}$).
When you substitute this into the induction equation, you get a **second derivative**:
[ \frac{\partial \mathbf{B}}{\partial t} \sim \nabla \times (\nabla \times \mathbf{B}) \sim \nabla^2 \mathbf{B} ]

This means the Hall term behaves mathematically like a **diffusion** term (parabolic), not an advection term (hyperbolic).

* **Ideal MHD** is like simulating sound waves (easy).
* **Hall MHD** is like simulating heat diffusion (hard), but worse, because it is "dispersive diffusion" that rotates the field vectors rather than just smoothing them.

#### Summary Comparison

| Feature                    | Ideal MHD                   | Hall MHD                         | Impact on Speed                            |
| :------------------------- | :-------------------------- | :------------------------------- | :----------------------------------------- |
| **Fastest Wave**           | Fast Magnetosonic           | Whistler Wave                    | Whistlers are much faster at grid scales.  |
| **Wave Speed ($v$)**       | Constant vs $\Delta x$      | Grows as $1 / \Delta x$          | Hall waves explode in speed on fine grids. |
| **Time Step ($\Delta t$)** | $\propto \Delta x$ (Linear) | $\propto \Delta x^2$ (Quadratic) | Hall requires drastically more steps.      |
| **Solver Effort**          | Simple Flux Calculation     | Matrix Inversion (Implicit)      | Hall steps are computationally heavy.      |
