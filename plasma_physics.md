This is a comprehensive technical breakdown of the three primary plasma physics models used in Field-Reversed Configuration (FRC) research, specifically aligned with the codes you mentioned: **Lamy Ridge** (Resistive MHD), **NIMROD** (Extended MHD), and **HYM** (Hybrid Kinetic).

---

# The Theoretical Hierarchy (Derivations)

All three models are derived from the **Two-Fluid Equations**, which treat ions ($i$) and electrons ($e$) as separate interpenetrating fluids coupled by Maxwell's equations.

**Common Ancestor Equations:**

* **Continuity:** $\partial_t n_\alpha + \nabla \cdot (n_\alpha \mathbf{v}_\alpha) = 0$
* **Momentum:** $m_\alpha n_\alpha d_t \mathbf{v}*\alpha = n*\alpha q_\alpha (\mathbf{E} + \mathbf{v}*\alpha \times \mathbf{B}) - \nabla p*\alpha + \mathbf{R}_{\alpha\beta}$
* **Maxwell:** $\partial_t \mathbf{B} = -\nabla \times \mathbf{E}$, $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$

---

## Model 1: Resistive MHD (The "Lamy Ridge" Model)

**Physics Level:** Macroscopic Fluid.
**Role:** Engineering design, circuit optimization, formation dynamics.

### 1.1 Derivation

* **Assumption 1 (Single Fluid):** Sum the ion and electron momentum equations. Neglect electron mass ($m_e \to 0$) and assume quasi-neutrality ($n_i \approx n_e$).
  [ \rho \frac{d\mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p ]
* **Assumption 2 (Simple Ohm’s Law):** Take the electron momentum equation. Neglect inertia terms and the Hall term ($\mathbf{J} \times \mathbf{B}$), assuming the macroscopic scale $L \gg d_i$ (ion skin depth).
  [ \mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} ]

### 1.2 Numerical Methods (Practical Implementation)

Lamy Ridge is typically a **2D Axisymmetric** code. This simplifies the math significantly.

* **Flux Function Formulation:** Instead of solving for vector $\mathbf{B}$, we solve for the poloidal flux $\psi(r,z)$.
  [ \mathbf{B} = \nabla \psi \times \nabla \phi + I \nabla \phi ]
* **The Grad-Shafranov Evolution:**
  [ \frac{\partial \psi}{\partial t} + \mathbf{v} \cdot \nabla \psi = \frac{\eta}{\mu_0} \Delta^* \psi + V_{loop} ]

  * **$\Delta^*$ Operator:** $\frac{\partial^2}{\partial r^2} - \frac{1}{r}\frac{\partial}{\partial r} + \frac{\partial^2}{\partial z^2}$.
* **Circuit Coupling:** The boundary condition for $\psi$ is not fixed. It is determined by the current $I_{coil}(t)$ in the external coils, which is solved simultaneously:
  [ V_{bank} = L_{coil} \frac{dI}{dt} + \frac{d}{dt} \int M_{plasma-coil} dI_{plasma} ]

### 1.3 FRC Adaptation & Critique

* **Adaptation:** **Chodura Resistivity**. Standard Spitzer resistivity is too low to model the rapid magnetic reconnection during FRC formation. A "anomalous" resistivity factor is added at the boundary ($\eta_{anom}$) to mimic micro-turbulence.
* **Critique:**

  * **Pros:** Extremely fast (minutes). Essential for tuning capacitor bank timing.
  * **Cons:** **Physics Failure.** Standard MHD predicts the FRC is unstable to the tilt mode on Alfvénic timescales. It cannot predict stability.

---

## Model 2: Extended MHD (The "NIMROD" Model)

**Physics Level:** Two-Fluid Fluid Dynamics.
**Role:** Global stability analysis, translation, transport scaling.

### 2.1 Derivation

* **Assumption:** We retain the single-fluid momentum equation but **keep the 2-fluid terms** in Ohm's Law.
* **Derivation:** Start with electron momentum ($m_e \to 0$) but do *not* drop the $\mathbf{J}$ terms.
  [ \mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J} + \underbrace{\frac{1}{ne}(\mathbf{J} \times \mathbf{B})}*{\text{Hall Term}} - \underbrace{\frac{1}{ne}\nabla p_e}*{\text{Electron Pressure}} ]

### 2.2 Numerical Methods (Semi-Implicit FEM)

NIMROD uses **High-Order Finite Elements** in the poloidal plane and **Fourier Decomposition** in the toroidal direction.

* **The Stiffness Problem:** The Hall term introduces **Whistler Waves**, where frequency $\omega \propto k^2$. As you refine the grid ($\Delta x \to 0$), the stable time step $\Delta t \to 0$ extremely fast.
* **The Solution: Semi-Implicit Time Stepping.**
  We modify the time advance of the magnetic field to "slow down" the fastest waves without affecting the large-scale plasma motion.
  [ \left( \mathbf{I} - \Delta t^2 L_{Hall} \right) \Delta \mathbf{B}^{n+1} = \text{Explicit Terms} ]
  Here, $L_{Hall}$ is a differential operator that dampens the high-$k$ whistler modes, allowing $\Delta t$ to be set by the slower Alfvén time.

### 2.3 FRC Adaptation & Critique

* **Adaptation:** **Vacuum Handling.** Extended MHD codes crash in vacuum ($n \to 0$). FRC simulations use a "Halo" of low-density, high-resistivity cold plasma outside the separatrix to model the vacuum region.
* **Critique:**

  * **Pros:** Captures the **Hall Stabilization** effect (separates electron/ion fluids), which correctly predicts that kinetic FRCs are tilt-stable.
  * **Cons:** Still a fluid model. It misses **Betatron Orbit** resonance, meaning it underestimates the stability provided by high-energy neutral beams.

---

## Model 3: Hybrid Kinetic-Fluid (The "HYM" Model)

**Physics Level:** Kinetic Ions + Fluid Electrons.
**Role:** The "Gold Standard" for stability and beam physics.

### 3.1 Derivation

* **Ions (Lagrangian):** Treated as particles to capture Finite Larmor Radius (FLR) effects.
  [ \frac{d\mathbf{v}_i}{dt} = \frac{q}{m_i} (\mathbf{E} + \mathbf{v}_i \times \mathbf{B}) ]
* **Electrons (Eulerian):** Treated as a massless fluid.
* **Coupling:** The Electric field is determined by the electron fluid equation, using the ion current $\mathbf{J}*i$ calculated from particles.
  [ \mathbf{E} = \frac{1}{en} (\mathbf{J}*{total} \times \mathbf{B} - \mathbf{J}_{i,kinetic} \times \mathbf{B}) - \frac{\nabla p_e}{en} + \eta \mathbf{J} ]

### 3.2 Numerical Methods (Delta-f PIC)

Standard Particle-In-Cell (PIC) is too noisy for macroscopic stability. HYM uses the **Delta-f ($\delta f$) Method**.

* **Concept:** We assume the distribution function is $f = f_0 + \delta f$, where $f_0$ is a known analytical equilibrium (e.g., Rigid Rotor).
* **Weight Equation:** Instead of creating particles from scratch, we simulate "markers" that carry a weight $w = \delta f / f$.
  [ \frac{dw}{dt} = -(1-w) \frac{d \ln f_0}{dt} ]
* **Benefit:** Noise is reduced by a factor of $1/\delta f$, allowing the code to resolve very small growth rates of instabilities.

### 3.3 FRC Adaptation & Critique

* **Adaptation:** **Linearized vs. Non-Linear Runs.**

  * *Linearized:* Used to check stability thresholds (e.g., "Is this FRC tilt stable?"). Very fast.
  * *Non-Linear:* Used to simulate turbulence and transport. Very slow.
* **Critique:**

  * **Pros:** The only model that accurately predicts FRC lifetime and beam stabilization.
  * **Cons:** **Computationally Expensive.** Cannot simulate the full formation-translation-merge-burn cycle. It is usually initialized from a Lamy Ridge or NIMROD snapshot.

---

# Summary Comparison for FRC Fusion

| Feature                | **Resistive MHD (Lamy Ridge)**                             | **Extended MHD (NIMROD)**                                                                          | **Hybrid Kinetic (HYM)**                                           |
| :--------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| **Primary Equations**  | $\mathbf{E} + \mathbf{v}\times\mathbf{B} = \eta\mathbf{J}$ | $\mathbf{E} + \mathbf{v}\times\mathbf{B} = \eta\mathbf{J} + \frac{\mathbf{J}\times\mathbf{B}}{ne}$ | $\mathbf{E} \leftarrow$ Fluid Closure; Ions $\leftarrow$ Particles |
| **Numerical Approach** | Finite Volume / 2D Grid                                    | Finite Element + Semi-Implicit                                                                     | Delta-f Particle-in-Cell                                           |
| **Computational Cost** | Low (Minutes)                                              | High (Hours/Days)                                                                                  | Extreme (Days/Weeks)                                               |
| **FRC Stability**      | **Fails** (Predicts instability)                           | **Good** (Captures Hall effect)                                                                    | **Excellent** (Captures FLR & Beams)                               |
| **Best Use Case**      | Circuit & Coil Design                                      | Global Dynamics & Thermal Transport                                                                | Stability Limits & NBI Physics                                     |
