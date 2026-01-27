Below is a **code-aligned** derivation + numerical-solver "cookbook" for three model families as they're commonly used for **FRC formation/translation/merging, pulsed compression, and sustainment**:

* **Resistive MHD + neutrals (Lamy Ridge–style)** (2-D, formation→merge, neutrals + ionization + radiation, wall geometry handling) ([jspf.or.jp][1])
* **Extended MHD / two-fluid (NIMROD–style)** (3-D, Hall + electron pressure + optional electron inertia; FE×Fourier; semi-implicit/implicit; atomic rates time-split) ([NIMROD团队][2])
* **Hybrid kinetic–fluid (HYM–style)** (full-orbit kinetic ions + fluid electrons; used to compare kinetic vs MHD for FRC merging/compression; boundary-driven compression) ([arXiv][3])

I'll do:

1. **equations + step-by-step derivation** (what's retained/dropped and why),
2. **numerical methods** under realistic FRC constraints,
3. **analysis/comparison/critiques** (what each model gets wrong in an FRC and how to mitigate).

---

# The Theoretical Hierarchy (Derivations)

All three models are derived from the **Two-Fluid Equations**, which treat ions ($i$) and electrons ($e$) as separate interpenetrating fluids coupled by Maxwell's equations.

**Common Ancestor Equations:**

* **Continuity:** $\partial_t n_\alpha + \nabla \cdot (n_\alpha \mathbf{v}_\alpha) = 0$
* **Momentum:** $m_\alpha n_\alpha d_t \mathbf{v}_\alpha = n_\alpha q_\alpha (\mathbf{E} + \mathbf{v}_\alpha \times \mathbf{B}) - \nabla p_\alpha + \mathbf{R}_{\alpha\beta}$
* **Maxwell:** $\partial_t \mathbf{B} = -\nabla \times \mathbf{E}$, $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$

---


## 1 Resistive MHD (Lamy Ridge–style): resistive MHD + energy + neutrals + atomic processes

The Lamy Ridge formulation is described publicly as: **2-D resistive MHD**, an **energy equation**, plus **neutral fluid equations**, with **momentum/density/energy exchange through ionization**, and non-adiabatic effects like **thermal conduction, ionization loss, radiation loss** ([jspf.or.jp][1]).

### 1.1 Step-by-step: from two-fluid → single-fluid resistive MHD

#### Step A: start with low-frequency Maxwell

$$
\partial_t\mathbf{B}=-\nabla\times\mathbf{E},\quad \nabla\cdot\mathbf{B}=0,\quad
\mathbf{J}=\frac{1}{\mu_0}\nabla\times\mathbf{B}.
$$

#### Step B: quasi-neutral single-fluid reduction

Assume quasi-neutrality and define bulk variables:

$$
\rho \approx m_i n,\quad \mathbf{u}\approx \mathbf{u}_i,\quad
\mathbf{J}=en(\mathbf{u}_i-\mathbf{u}_e).
$$

#### Step C: continuity (sum mass-weighted species continuity)

$$
\boxed{\partial_t \rho+\nabla\cdot(\rho\mathbf{u}) = S_\rho}
$$

#### Step D: momentum (sum ion+electron momentum)

Electric forces cancel under quasi-neutrality; Lorentz remains:

$$
\boxed{
\partial_t(\rho\mathbf{u})+\nabla\cdot(\rho\mathbf{u}\mathbf{u}+p\mathbf{I})
=\mathbf{J}\times\mathbf{B}+\nabla\cdot\boldsymbol{\Pi}+\mathbf{S}_m
}
$$

#### Step E: Ohm's law (resistive closure)

Start from electron momentum, drop electron inertia and electron pressure terms (this is the key resistive-MHD simplification), and model collisional friction as $\eta\mathbf{J}$:

$$
\boxed{\mathbf{E}+\mathbf{u}\times\mathbf{B}=\eta \mathbf{J}}
$$

#### Step F: induction (plug Ohm's law into Faraday)

$$
\boxed{
\partial_t\mathbf{B}=\nabla\times(\mathbf{u}\times\mathbf{B})-\nabla\times(\eta\mathbf{J})
}
$$

#### Step G: energy (numerically robust form)

Lamy Ridge is explicitly described as having an energy equation with non-adiabatic effects ([jspf.or.jp][1]), so the numerically stable "core" is conservative total energy:

$$
E=\frac{p}{\gamma-1}+\frac{1}{2}\rho u^2+\frac{B^2}{2\mu_0}
$$

$$
\boxed{
\partial_tE+\nabla\cdot\left[\left(E+p+\frac{B^2}{2\mu_0}\right)\mathbf{u}-\frac{(\mathbf{u}\cdot\mathbf{B})\mathbf{B}}{\mu_0}\right]
=\eta J^2-\nabla\cdot\mathbf{q}-P_{\rm rad}-P_{\rm ion}+S_E
}
$$

where $P_{\rm ion}$ represents ionization energy sinks, and $\mathbf{q}$ includes thermal conduction.

### 1.2 Neutral-fluid coupling (the "Lamy Ridge" differentiator)

A minimal neutral subsystem consistent with the public description ([jspf.or.jp][1]):

**Neutral mass**

$$
\partial_t \rho_n+\nabla\cdot(\rho_n\mathbf{u}_n)= -S_{\rm ion}+S_{\rm rec}
$$

**Neutral momentum**

$$
\partial_t(\rho_n\mathbf{u}_n)+\nabla\cdot(\rho_n\mathbf{u}_n\mathbf{u}_n+p_n\mathbf{I})
= -\mathbf{R}_{\rm ion}-\mathbf{R}_{\rm cx}
$$

**Neutral energy**

$$
\partial_t E_n+\nabla\cdot(\cdots)= -Q_{\rm ion}-Q_{\rm cx}+\cdots
$$

Plasma gets the equal-and-opposite sources:

$$
S_\rho = +S_{\rm ion}-S_{\rm rec},\quad
\mathbf{S}_m=+\mathbf{R}_{\rm ion}+\mathbf{R}_{\rm cx},\quad
S_E=+Q_{\rm ion}+Q_{\rm cx}.
$$

**Why this matters for FRC formation/merge:** at mTorr fills, neutrals and CX can strongly modify momentum balance, shock heating, and radiation—so this model is often a better *formation/translation* predictor than "plasma-only" MHD.

---

### 1.3 Practical numerics for Lamy Ridge–type FRC runs (2-D)

**Best-fit assumptions:** axisymmetric $(r)$–$(z)$, strong pulsed sources, complicated vessel geometry, strong non-ideal sources.

**Spatial discretization**

* Finite-volume Godunov (HLLD-like) for the hyperbolic (ideal) part.
* Axisymmetric source-term handling (geometric terms) must be well-balanced.

**Maintain $\nabla\cdot \mathbf{B}=0$**

* Prefer **constrained transport**. In FRCs, divergence errors near the null/separatrix quickly pollute reconnection and pressure balance.

**Operator splitting (IMEX)**

* Explicit: ideal MHD advection.
* Implicit: resistive diffusion and thermal conduction (diffusive CFL is brutal).
* Source ODEs for ionization/radiation: often stiff → integrate with subcycling or implicit/analytic updates.

**Geometry**

* Lamy Ridge is described as handling complex boundaries (e.g., cut-cell style in public descriptions elsewhere), which pairs naturally with FV. ([jspf.or.jp][1])

**Critique**

* Resistive MHD reconnection is "forced" by $\eta$, so merging-layer physics and ion heating partition can be wrong (qualitatively, not just quantitatively).
* No ion-orbit physics at the null (a big deal for FRCs).

---

## 2 Extended MHD (NIMROD–style): Hall + electron pressure + optional electron inertia; FE×Fourier; implicit time advance

NIMROD explicitly states it solves **fully 3-D extended MHD** using **spectral finite elements (2-D)** + **Fourier (3rd dimension)** with **semi-implicit and implicit time discretization**; it also time-splits atomic rates like ionization/radiation, and supports kinetic closures / energetic particle options. ([NIMROD团队][2])

### 2.1 Step-by-step: generalized Ohm's law (what makes it "extended")

#### Step A: electron momentum

$$
m_e n(\partial_t\mathbf{u}_e+\mathbf{u}_e\cdot\nabla\mathbf{u}_e)
=-en(\mathbf{E}+\mathbf{u}_e\times\mathbf{B})-\nabla\cdot\mathbf{P}_e+\mathbf{R}_{ei}.
$$

#### Step B: divide by $(-en)$ and model friction

Let $\mathbf{R}_{ei}/(en)=\eta\mathbf{J}$.

$$
\mathbf{E}+\mathbf{u}_e\times\mathbf{B}
=-\frac{1}{en}\nabla\cdot\mathbf{P}_e+\eta\mathbf{J}
-\frac{m_e}{e}(\partial_t\mathbf{u}_e+\mathbf{u}_e\cdot\nabla\mathbf{u}_e).
$$

#### Step C: replace $\mathbf{u}_e$ using $\mathbf{J}=en(\mathbf{u}_i-\mathbf{u}_e)$

With $\mathbf{u}\approx\mathbf{u}_i$, $\mathbf{u}_e=\mathbf{u}-\mathbf{J}/(en)$, so

$$
\mathbf{u}_e\times\mathbf{B}=\mathbf{u}\times\mathbf{B}-\frac{\mathbf{J}\times\mathbf{B}}{en}.
$$

#### Step D: obtain generalized Ohm's law

$$
\boxed{
\mathbf{E}+\mathbf{u}\times\mathbf{B}
=\eta\mathbf{J}
+\frac{\mathbf{J}\times\mathbf{B}}{en}
-\frac{1}{en}\nabla\cdot\mathbf{P}_e
-\frac{m_e}{e}(\partial_t\mathbf{u}_e+\mathbf{u}_e\cdot\nabla\mathbf{u}_e)
}
$$

Common practical reductions (often used in extended-MHD codes):

* $\nabla\cdot\mathbf{P}_e \to \nabla p_e$ (isotropic electrons),
* electron inertia reduced to $(m_e/ne^2)\partial_t\mathbf{J}$ if retained.

Then induction is still:

$$
\partial_t\mathbf{B}=-\nabla\times\mathbf{E},\qquad \mathbf{J}=\nabla\times\mathbf{B}/\mu_0.
$$

#### Step E: the rest of the system

The continuity and momentum equations are similar to resistive MHD, but **two-temperature energy** and **anisotropic transport** are typically essential for FRC realism (ends + radiation). NIMROD also time-splits atomic-rate effects like ionization/radiation. ([NIMROD团队][2])

---

### 2.2 Practical numerics for NIMROD-style extended MHD (FRC-relevant)

#### Space: FE in poloidal plane × Fourier in toroidal/3rd dimension

This is explicitly part of NIMROD's approach. ([NIMROD团队][2])
Why it works well for FRCs:

* Axisymmetric base equilibrium + low-$n$ 3-D modes (tilt/shift) are naturally represented.
* High-order FE helps with smooth global modes and avoids excessive numerical diffusion.

#### Time: semi-implicit / implicit is not optional

Hall introduces dispersive whistler waves with nasty explicit CFL. NIMROD emphasizes semi-implicit/implicit temporal discretization for fusion time scales. ([NIMROD团队][2])

A realistic IMEX split:

* **Explicit**: ideal advection parts (hyperbolic).
* **Implicit**: Hall term, resistive diffusion, anisotropic conduction, stiff radiation sinks.

#### Linear/nonlinear solves and preconditioning

Extended MHD implicit steps require good preconditioners for:

* diffusion operators,
* curl–curl blocks,
* coupled Hall + pressure terms.

#### FRC boundary conditions (the usual failure point)

To be credible for FRC *fusion* (not just MHD benchmarks), you must model:

* conducting wall and/or resistive wall,
* open ends + mirror coils (or equivalently, end-loss models).
  If you don't, extended MHD will tend to overpredict confinement and stability.

**Critique**

* Extended MHD still misses genuine ion-orbit physics at the null unless augmented with FLR/kinetic closures.
* Numerical stiffness + closure sensitivity means "pretty plots" can hide large modeling error if boundaries/transport aren't realistic.

---

## 3 Hybrid kinetic–fluid (HYM–style): kinetic ions, fluid electrons; used for FRC merging/compression comparisons

HYM is described as supporting resistive MHD and hybrid models with **ions as particles and electrons as a fluid** for FRC studies. ([pppl.gov][4])
Recent work reports **2-D hybrid (fluid electrons + full-orbit kinetic ions)** simulations of **FRC merging and compression** using HYM, comparing kinetic vs MHD results. ([arXiv][3])

### 3.1 Step-by-step derivation: hybrid Ohm's law (electron-fluid closure)

#### Step A: kinetic ions

$$
\boxed{
\partial_t f_i+\mathbf{v}\cdot\nabla f_i+\frac{q_i}{m_i}(\mathbf{E}+\mathbf{v}\times\mathbf{B})\cdot\nabla_{\mathbf{v}}f_i
= C[f_i]+S_i
}
$$

Moments:

$$
n=\int f_i\,d^3v,\quad \mathbf{u}_i=\frac{1}{n}\int \mathbf{v} f_i\,d^3v.
$$

#### Step B: massless electron momentum (fluid electrons)

$$
0=-en(\mathbf{E}+\mathbf{u}_e\times\mathbf{B})-\nabla p_e+\eta\mathbf{J}.
$$

So

$$
\mathbf{E}=-\mathbf{u}_e\times\mathbf{B}-\frac{\nabla p_e}{en}+\eta\mathbf{J}.
$$

#### Step C: eliminate $\mathbf{u}_e$ using $\mathbf{J}=en(\mathbf{u}_i-\mathbf{u}_e)$

$$
\mathbf{u}_e=\mathbf{u}_i-\frac{\mathbf{J}}{en}
$$

$$
\boxed{
\mathbf{E}= -\mathbf{u}_i\times\mathbf{B}
+\frac{\mathbf{J}\times\mathbf{B}}{en}
-\frac{\nabla p_e}{en}
+\eta\mathbf{J}
}
$$

#### Step D: field update

$$
\partial_t\mathbf{B}=-\nabla\times\mathbf{E},\qquad \mathbf{J}=\nabla\times\mathbf{B}/\mu_0.
$$

#### Boundary drive for compression (FRC-relevant detail)

The recent HYM FRC paper applies compression through **time-dependent boundary conditions for the toroidal vector potential $A_\phi$**. ([arXiv][3])
That's a common hybrid technique: it injects the correct inductive drive without explicitly meshing external coils.

---

### 3.2 Practical numerics for HYM-style hybrid (what actually runs)

A standard, stable loop:

1. **Particle push** (Boris) with $\mathbf{E}^n,\mathbf{B}^n$.
2. **Deposit** $n,\mathbf{u}_i$ on grid (use charge-conserving schemes if possible).
3. Compute $\mathbf{J}=\nabla\times\mathbf{B}/\mu_0$.
4. Compute $\mathbf{E}$ from hybrid Ohm's law.
5. Update $\mathbf{B}^{n+1}=\mathbf{B}^n-\Delta t\,\nabla\times\mathbf{E}$ (CT-like update preferred).
6. Apply collisions/sources + boundary conditions (including $A_\phi(t)$ drive). ([arXiv][3])

**FRC practical constraints**

* Compression increases $\Omega_{ci}$ and gradients → timestep constraints tighten.

  * Common fix: particle subcycling; semi-implicit field solves.
* PIC noise can contaminate $\nabla p_e$ and hence $\mathbf{E}$.

  * Fix: more particles/cell; careful smoothing that preserves Hall physics; quiet starts.

**Critique**

* Hybrid is the best of these three for null-region orbit physics and merging sensitivity, but it's expensive for full 3-D reactor pulse trains.
* Electron kinetics (Landau damping, electron FLR) are not captured unless you extend the electron model.

---

# Comparison and "which model is best for which FRC fusion question"

### Predictive capability for FRC fusion-relevant phenomena

**Merging / reconnection / ion heating partition**

* Best: **HYM-style hybrid** (ion orbits + Hall naturally) ([arXiv][3])
* Risky: resistive MHD (reconnection forced via $\eta$)

**3-D stability (tilt/shift) during pulsed compression**

* Best: **NIMROD-style extended MHD** (3-D FE×Fourier + implicit) ([NIMROD团队][2])
* Hybrid is possible but expensive; resistive MHD can miss key dispersive physics.

**Transport to open ends / mirror effects / wall coupling**

* Extended MHD or resistive MHD can do it *if* boundary models are realistic.
* Biggest gotcha: "nice confinement" often comes from overly ideal end boundary conditions, not physics.

**Fast-ion confinement (NBI, D–³He protons, alpha heating)**

* Needs hybrid energetic particle capability (either full hybrid or "fluid bulk + kinetic minority").
* Pure fluid MHD (even extended) often cannot reliably predict orbit loss fractions.

---

## Bottom-line critiques (what you should be skeptical of)

* **Resistive MHD (Lamy Ridge)**: great for integrated formation + neutrals + radiation, but fundamentally limited for kinetic reconnection and orbit physics; treat it as *engineering/formation* fidelity. ([jspf.or.jp][1])
* **Extended MHD (NIMROD)**: best "global 3-D workhorse," but results are only as good as closures (transport, end BCs, resistivity, wall model). Its implicit machinery is essential because the physics is stiff. ([NIMROD团队][2])
* **Hybrid (HYM)**: strongest for merging/orbit physics, but cost/noise and electron-kinetic omission require careful interpretation; boundary-driven compression via $A_\phi(t)$ is powerful but you must validate it against coil-coupled cases. ([arXiv][3])


## Summary Comparison for FRC Fusion

| Feature                | **Resistive MHD (Lamy Ridge)**                             | **Extended MHD (NIMROD)**                                                                          | **Hybrid Kinetic (HYM)**                                           |
| :--------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| **Primary Equations**  | $\mathbf{E} + \mathbf{v}\times\mathbf{B} = \eta\mathbf{J}$ | $\mathbf{E} + \mathbf{v}\times\mathbf{B} = \eta\mathbf{J} + \frac{\mathbf{J}\times\mathbf{B}}{ne}$ | $\mathbf{E} \leftarrow$ Fluid Closure; Ions $\leftarrow$ Particles |
| **Numerical Approach** | Finite Volume / 2D Grid                                    | Finite Element + Semi-Implicit                                                                     | Delta-f Particle-in-Cell                                           |
| **Computational Cost** | Low (Minutes)                                              | High (Hours/Days)                                                                                  | Extreme (Days/Weeks)                                               |
| **FRC Stability**      | **Fails** (Predicts instability)                           | **Good** (Captures Hall effect)                                                                    | **Excellent** (Captures FLR & Beams)                               |
| **Best Use Case**      | Circuit & Coil Design                                      | Global Dynamics & Thermal Transport                                                                | Stability Limits & NBI Physics                                     |


---


[1]: https://www.jspf.or.jp/PFR/PDF2020/pfr2020_15-2402020.pdf?utm_source=chatgpt.com "Plasma and Fusion Research,ISSN 1880-6821 - jspf.or.jp"
[2]: https://nimrodteam.org/?utm_source=chatgpt.com "NIMROD Magnetohydrodynamic Code Team Homepage — nimrodteam.org"
[3]: https://arxiv.org/html/2501.03425v1?utm_source=chatgpt.com "Hybrid simulations of FRC merging and compression - arXiv.org"
[4]: https://www.pppl.gov/research/theory/codes?utm_source=chatgpt.com "Codes | Princeton Plasma Physics Laboratory"
