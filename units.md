## The importance of dimensionless units in numerical MHD solvers

---

### **1. Motivation: Why Nondimensionalize?**

In numerical solvers, working with raw SI (System International) or CGS (Gaussian) units is rarely done. Instead, equations are cast into a dimensionless form. This serves three critical purposes:

1. **Numerical Precision (Floating Point Arithmetic):**
   Real physical values can vary wildly in magnitude (e.g., density (\rho \sim 10^{-20}) kg/m(^3) in space vs. (\rho \sim 10^3) kg/m(^3) in liquid metals). Computers use floating-point arithmetic (IEEE 754), which has limited precision. multiplying extremely small numbers by extremely large ones leads to round-off errors. Scaling variables to be Order of Magnitude 1 ((O(1))) preserves precision.
2. **Universality:**
   A single simulation with dimensionless parameters can describe completely different physical systems. For example, a simulation with Reynolds number (Re=1000) applies equally to a small pipe in a lab and a massive river, provided the geometric scaling matches.
3. **Parameter Reduction:**
   It isolates the *ratios* of forces that actually govern the physics, reducing the number of free parameters.

---

### **2. The Normalization Procedure**

To nondimensionalize, we choose **Characteristic Scales** (base units) for the fundamental quantities. A common choice in MHD is:

* **Length Scale ((L_0)):** The size of the system (e.g., box size, radius).
* **Density Scale ((\rho_0)):** The average initial density.
* **Velocity Scale ((V_0)):** Often chosen as the **Alfvén speed** (V_A = B_0 / \sqrt{\mu_0 \rho_0}) or the sound speed (C_s).
* **Magnetic Field Scale ((B_0)):** The average background field strength.

From these, we derive scales for time ((t_0 = L_0/V_0)) and pressure ((P_0 = \rho_0 V_0^2)).

The variables in the code ((\tilde{x}, \tilde{t}, \tilde{\mathbf{u}})) are related to physical values by:
[ x = L_0 \tilde{x}, \quad t = t_0 \tilde{t}, \quad \mathbf{u} = V_0 \tilde{\mathbf{u}} ]

When these are substituted back into the MHD equations, physical constants like (\mu_0) drop out, replaced by dimensionless numbers.

---

### **3. Key Dimensionless Numbers in Solvers**

These numbers appear as coefficients in the normalized equations (e.g., (1/Re), (1/R_m)) and dictate the "regime" of the simulation.

#### **A. Reynolds Number ((Re))**

[ Re = \frac{\rho_0 V_0 L_0}{\mu_{visc}} = \frac{V_0 L_0}{\nu} ]

* **Ratio:** Inertial Forces / Viscous Forces.
* **Numerical Context:**

  * **High (Re):** Turbulent flow. Requires high resolution to resolve small eddies.
  * **Low (Re):** Laminar, viscous flow.
  * **Solver Note:** If (Re \to \infty) (Euler equations), the solver relies on "numerical viscosity" to stabilize shocks.

#### **B. Magnetic Reynolds Number ((R_m))**

[ R_m = \frac{V_0 L_0}{\eta} ]

* **Ratio:** Magnetic Advection / Magnetic Diffusion.
* **Numerical Context:**

  * **High (R_m):** Ideal MHD. Field is frozen in. Requires flux-conserving schemes (Frozen Flux problem).
  * **Low (R_m):** Resistive MHD. Field diffuses rapidly. Requires implicit time-stepping or very small time steps (Diffusion problem).

#### **C. Lundquist Number ((S))**

[ S = \frac{L_0 V_A}{\eta} ]

* **Definition:** The ratio of the resistive diffusion time to the Alfvén crossing time. It is effectively (R_m) when the velocity scale is explicitly the Alfvén speed.
* **Numerical Context:** Critical in **Reconnection** studies. High (S) simulations (e.g., (S=10^6)) are notoriously difficult because current sheets become extremely thin ((\delta \sim L_0 / \sqrt{S})), requiring Adaptive Mesh Refinement (AMR).

#### **D. Plasma Beta ((\beta))**

[ \beta = \frac{P_{thermal}}{P_{magnetic}} = \frac{2\mu_0 P}{B^2} ]

* **Ratio:** Thermal Pressure / Magnetic Pressure.
* **Numerical Context:**

  * **Low (\beta) ((\ll 1)):** Magnetically dominated (e.g., Solar Corona). The magnetic field controls the dynamics. Forces are stiff; errors in calculating (\mathbf{B}) can catastrophically disrupt the gas pressure.
  * **High (\beta) ((\gg 1)):** Hydrodynamically dominated (e.g., Solar Interior). The field is passively dragged by the fluid.

#### **E. Alfvénic Mach Number ((M_A))**

[ M_A = \frac{V_{flow}}{V_A} ]

* **Ratio:** Flow velocity / Alfvén wave speed.
* **Numerical Context:** Determines the nature of shocks.

  * (M_A < 1): Sub-Alfvénic flow. Waves can propagate upstream (boundary conditions must allow this).
  * (M_A > 1): Super-Alfvénic flow. Shocks form; no information propagates upstream (inflow boundaries can be fixed).

---

### **4. The Dimensionless MHD Equations**

In a code, the equations often look like this (assuming units where (L_0=1, \rho_0=1, V_0=1)):

**Momentum Equation:**
[
\frac{\partial \tilde{\mathbf{u}}}{\partial \tilde{t}} + (\tilde{\mathbf{u}} \cdot \tilde{\nabla}) \tilde{\mathbf{u}} = -\tilde{\nabla} \tilde{P} + (\tilde{\nabla} \times \tilde{\mathbf{B}}) \times \tilde{\mathbf{B}} + \frac{1}{Re} \tilde{\nabla}^2 \tilde{\mathbf{u}}
]
*(Notice (\mu_0) is gone. If (Re) is high, the last term is ignored).*

**Induction Equation:**
[
\frac{\partial \tilde{\mathbf{B}}}{\partial \tilde{t}} = \tilde{\nabla} \times (\tilde{\mathbf{u}} \times \tilde{\mathbf{B}}) + \frac{1}{R_m} \tilde{\nabla}^2 \tilde{\mathbf{B}}
]

---

### **5. Implications for Numerical Stability (CFL Condition)**

Dimensionless numbers directly determine the maximum allowable Time Step ((\Delta t)) in explicit solvers via the Courant–Friedrichs–Lewy (CFL) condition.

The time step is limited by the **fastest wave speed** in the grid. In MHD, this is the **Fast Magnetosonic Wave** ((V_f)):
[ V_f^2 = V_A^2 + C_s^2 ]

In dimensionless units, the constraint is:
[ \Delta \tilde{t} < C_{CFL} \frac{\Delta \tilde{x}}{\sqrt{\tilde{V}_A^2 + \tilde{C}_s^2}} ]

* **Low (\beta) Problem:** If (\beta) is very low, (V_A) becomes huge (since (B) is large and (\rho) is small). This makes (\Delta t) tiny, making the simulation very expensive computationally.
* **Low (R_m) Problem:** If calculating diffusion explicitly, the constraint is (\Delta t \sim \Delta x^2 R_m). For low (R_m) (high resistivity), this is very restrictive ((\Delta x^2) is much smaller than (\Delta x)).

### **Summary**

| Dimensionless Unit  | Interpretation              | Solver Challenge                                                   |
| :------------------ | :-------------------------- | :----------------------------------------------------------------- |
| **(R_m)**           | Advection vs. Diffusion     | High (R_m): Oscillation/Dispersion. Low (R_m): Stiff time step.    |
| **(\beta)**         | Gas vs. Mag. Pressure       | Low (\beta): Tiny time steps (Fast waves), high error sensitivity. |
| **(S) (Lundquist)** | Ideal Time / Resistive Time | High (S): Needs massive grid resolution for reconnection layers.   |
| **(M_A)**           | Flow vs. Alfvén Speed       | Determining inflow/outflow boundary behavior (Shocks).             |

## Practical guidelines for handling constants and units when developing or setting up an MHD solver

### **1. The Golden Rule: Code in "Unity"**

**Never hardcode physical constants** (like `1.67e-27` for proton mass or `4*pi*1e-7` for $\mu_0$) directly into your core solver loops.

* **Practice:** Write your flux loops and source terms assuming all characteristic scales are **1.0**.
* **Why:** This decouples your numerical algorithm from the specific physics problem. A solver that calculates $\partial_t \mathbf{B} = \nabla \times (\mathbf{u} \times \mathbf{B})$ works equally well for a galaxy and a tokamak; only the input configuration file changes.

---

### **2. Handling the $\mu_0$ / $\sqrt{4\pi}$ Nightmare**

The most common bug in MHD codes arises from the definition of magnetic pressure and the conversion between SI and CGS (Gaussian) units.

* **The Problem:**

  * SI: $P_{mag} = B^2 / (2\mu_0)$
  * CGS: $P_{mag} = B^2 / (8\pi)$
  * Heaviside-Lorentz: $P_{mag} = B^2 / 2$
* **Practical Advice:** Define a **Modified Magnetic Field** in your code:
  [ \mathbf{B}*{code} = \frac{\mathbf{B}*{physical}}{\sqrt{\mu_0}} \quad (\text{SI}) \quad \text{or} \quad \mathbf{B}*{code} = \frac{\mathbf{B}*{physical}}{\sqrt{4\pi}} \quad (\text{CGS}) ]
  If you do this, the momentum equation source term always simplifies to:
  [ \text{Force} = (\nabla \times \mathbf{B}*{code}) \times \mathbf{B}*{code} ]
  and magnetic pressure is simply:
  [ P_{mag, code} = \frac{1}{2} B_{code}^2 ]
  **Benefit:** Your code becomes agnostic to the unit system. You handle the $\sqrt{4\pi}$ or $\mu_0$ factors *only* during Input (reading initial conditions) and Output (post-processing).

---

### **3. Controlling the Time Step (The "Vacuum" Problem)**

In MHD, the time step $\Delta t$ is inversely proportional to the Alfvén speed $V_A = B / \sqrt{\rho}$.

* **The Trap:** In low-density regions (like the corner of a simulation box representing "vacuum" outside a star), $\rho \to 0$ implies $V_A \to \infty$. This will crash your simulation or drive $\Delta t$ to zero, stalling the run.
* **Practical Fixes:**

  1. **Density Floor:** Enforce a strict minimum density $\rho_{min}$ (e.g., $10^{-6} \rho_0$). Never allow the code to generate negative density or vacuum.
  2. **Alfvén Speed Limiter:** Artificially limit the characteristic speed in the CFL calculation.
     [ V_{char} = \sqrt{V_{sound}^2 + \min(V_A^2, V_{max}^2)} ]
     *Warning: This alters physics in the low-density region. Ensure this region is not the focus of your study.*

---

### **4. Choosing Your Characteristic Scales ($L_0, \rho_0, V_0$)**

Don't choose these arbitrarily. Choose them to maximize floating-point precision ($O(1)$) for the **most important** part of your domain.

* **Scenario A: Supersonic Astrophysics (Supernovae, Jets)**

  * **Priority:** Capturing shocks.
  * $V_0$: Choose the shock speed or average flow speed.
  * $L_0$: Radius of the jet/star.
  * $P_0$: $\rho_0 V_0^2$ (Dynamic pressure dominant).
* **Scenario B: Magnetic Confinement (Tokamaks, Coronal Loops)**

  * **Priority:** Magnetic equilibrium.
  * $V_0$: Choose the Alfvén speed $V_A$.
  * $P_0$: $B_0^2$ (Magnetic pressure dominant).
  * *Tip:* If you choose sound speed for $V_0$ in a low-$\beta$ problem, your magnetic field values will be huge (e.g., $B \sim 100$), leading to precision loss when calculating $B^2 - P_{gas}$.

---

### **5. The "Grid" Reynolds Number check**

You might set a physical viscosity $\nu$ to achieve $Re=1000$. However, your grid resolution restricts the *effective* Reynolds number you can actually simulate.

* **The Limit:** The Numerical Reynolds Number is roughly:
  [ Re_{num} \approx \frac{L}{ \Delta x } \times (\text{Order of Accuracy}) ]
* **Advice:** Calculate the **Cell Peclet Number** (or Cell Reynolds Number) in your code diagnostics:
  [ P_{cell} = \frac{|u| \Delta x}{\nu} ]

  * If $P_{cell} \gg 1$: Your explicit physical viscosity is negligible compared to the numerical truncation error. You are running an "Implicit Large Eddy Simulation" (ILES), not a resolved DNS.
  * If you need specific physical diffusion (e.g., matching a lab experiment), you must refine the grid until $P_{cell} \sim O(1)$.

---

### **6. Post-Processing: Converting Back to Reality**

When writing a paper, you must convert your code units (C.U.) back to physical units (P.U.). Keep a "units.txt" log file for every run.

**Conversion Map:**

| Quantity          | Physical Unit (P.U.) | Calculation from Code Unit (C.U.)                  |
| :---------------- | :------------------- | :------------------------------------------------- |
| **Time**          | seconds              | $t_{PU} = t_{CU} \times (L_0 / V_0)$               |
| **Velocity**      | m/s                  | $u_{PU} = u_{CU} \times V_0$                       |
| **Density**       | kg/m³                | $\rho_{PU} = \rho_{CU} \times \rho_0$              |
| **Pressure**      | Pascals              | $P_{PU} = P_{CU} \times (\rho_0 V_0^2)$            |
| **B-Field (SI)**  | Tesla                | $B_{PU} = B_{CU} \times (V_0 \sqrt{\mu_0 \rho_0})$ |
| **B-Field (CGS)** | Gauss                | $B_{PU} = B_{CU} \times (V_0 \sqrt{4\pi \rho_0})$  |

### **Summary Checklist for Developers**

1. [ ] **Normalize Inputs:** Have a wrapper that reads physical inputs and divides them by $L_0, \rho_0, V_0$ before the solver starts.
2. [ ] **Unit-Agnostic Core:** Ensure no $\pi$ or $\mu_0$ appears in the update equations.
3. [ ] **Vacuum Protection:** Implement density and pressure floors.
4. [ ] **Output Metadata:** Save the values of $L_0, \rho_0, V_0$ in the output file header (e.g., HDF5 attributes) so you never forget what "Time = 10.0" actually means years later.
