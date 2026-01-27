## 1) Universal fusion-burn network (any fuel set)

Consider ion species (s \in {D, T,, ^3!He,, p,, ^{11}!B,\dots}) with densities (n_s), ion temperature (T_i), electron temperature (T_e).

### 1.1 Reaction rates

For a binary reaction (a+b\to\cdots), the **volumetric reaction rate**
[
R_{ab} ;=; \frac{n_a n_b}{1+\delta_{ab}};\langle \sigma v\rangle_{ab}(T_i)
]
((\delta_{ab}=1) if (a=b), else 0).

### 1.2 Species ODEs (0D burn / “box” model)

Let (\nu_{s,ab}) be the **net number of ions of species (s)** created per reaction (ab) (negative if consumed). Then
[
\frac{dn_s}{dt} ;=; \sum_{ab}\nu_{s,ab},R_{ab} ;-; \frac{n_s}{\tau_{p,s}}
]
where (\tau_{p,s}) is an effective particle loss / exhaust time (in your pulsed FRC, this is “small” during the burn window and then deliberately large during exhaust clearing).

### 1.3 Fusion power and partition into fast products

For each reaction (ab) with Q-value (Q_{ab}),
[
P_{\text{fus}} = \sum_{ab} R_{ab},Q_{ab}
]
Split into **charged** vs **neutral (neutron/gamma)** channels:
[
P_{\text{ch}} = \sum_{ab} R_{ab},Q_{ab},f_{\text{ch},ab},\qquad
P_{\text{neu}} = \sum_{ab} R_{ab},Q_{ab},(1-f_{\text{ch},ab})
]
In aneutronic fuels, (f_{\text{ch}}\approx 1), which is why your concept emphasizes D–(^3)He and direct conversion .

---

## 2) Fuel-specific burn equations (DT, D–(^3)He, p–(^\text{11})B, DD+secondaries)

### 2.1 Deuterium–Tritium (DT)

Reaction: (D+T\to \alpha(3.5\text{ MeV})+n(14.1\text{ MeV})).

Rate:
[
R_{DT}=n_D n_T \langle\sigma v\rangle_{DT}(T_i)
]
Species:
[
\frac{dn_D}{dt}=-R_{DT},\quad \frac{dn_T}{dt}=-R_{DT}
]
Power:
[
P_{\text{fus}}=R_{DT}Q_{DT},\quad Q_{DT}=17.6\text{ MeV}
]
Charged vs neutron:
[
P_{\text{ch}}=R_{DT},3.5\text{ MeV},\quad P_{\text{neu}}=R_{DT},14.1\text{ MeV}
]
**Implication for energy recovery:** inductive/direct conversion can only “see” the part that ends up as **plasma pressure / magnetic work** (ultimately sourced from alpha deposition); neutron power is mostly unrecoverable electrically.

---

### 2.2 Deuterium–Helium-3 (D–(^3)He)

Reaction: (D+^3!He\to \alpha(3.6\text{ MeV})+p(14.7\text{ MeV})) (charged products).

Rate:
[
R_{D3}=n_D n_{3}\langle\sigma v\rangle_{D3}(T_i)
]
Species:
[
\frac{dn_D}{dt}=-R_{D3},\qquad \frac{dn_{3}}{dt}=-R_{D3}
]
Power (mostly charged):
[
P_{\text{ch}}\approx P_{\text{fus}}=R_{D3}Q_{D3},\quad Q_{D3}\approx 18.3\text{ MeV}
]
**Side reactions:** any deuterium plasma also has DD reactions, producing neutrons and tritium; in your design doc you already note D–D side neutrons and shielding needs . So in practice you model **D–(^3)He + DD network** (see §2.4).

---

### 2.3 Proton–Boron-11 (p–(^\text{11})B)

Reaction: (p+^{11}!B \to 3\alpha) (charged products).

Rate:
[
R_{pB}=n_p n_B \langle\sigma v\rangle_{pB}(T_i)
]
Species:
[
\frac{dn_p}{dt}=-R_{pB},\qquad \frac{dn_B}{dt}=-R_{pB}
]
Power:
[
P_{\text{ch}}\approx P_{\text{fus}}=R_{pB}Q_{pB},\quad Q_{pB}\approx 8.7\text{ MeV}
]
**Key coupling:** p–B(^\text{11})B is often limited not by the burn ODEs but by the **electron energy balance** (bremsstrahlung scales strongly with (Z) and (T_e)), so you must carry a **two-temperature** model ((T_i \neq T_e)) in §3.

---

### 2.4 Deuterium–Deuterium (DD) with secondaries (DT and D–(^3)He)

Primary DD branches (roughly comparable):

* (D+D\to T + p)
* (D+D\to, ^3!He + n)

Define
[
R_{DD}=\frac{1}{2}n_D^2\langle\sigma v\rangle_{DD}(T_i)
]
With branch fractions (b_{Tp}), (b_{3n}) (often (\approx 0.5) each; keep symbolic). Then:
[
\frac{dn_D}{dt}=-2R_{DD} ;-;R_{DT};-;R_{D3}
]
[
\frac{dn_T}{dt}=+b_{Tp}R_{DD};-;R_{DT}
]
[
\frac{dn_{3}}{dt}=+b_{3n}R_{DD};-;R_{D3}
]
where the secondary rates are
[
R_{DT}=n_D n_T\langle\sigma v\rangle_{DT}(T_i),\qquad
R_{D3}=n_D n_3\langle\sigma v\rangle_{D3}(T_i)
]
Total power:
[
P_{\text{fus}}=R_{DD}Q_{DD}+R_{DT}Q_{DT}+R_{D3}Q_{D3}
]
and similarly partition charged vs neutrons branch-by-branch.

---

## 3) Tie-in to MHD: where burn enters the FRC equations

A practical reactor model is usually **two-fluid thermodynamics** (ions/electrons) + single-fluid momentum + Maxwell.

### 3.1 Resistive MHD (baseline)

Continuity:
[
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0
]
Momentum:
[
\rho\left(\frac{\partial \mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)
= -\nabla p + \mathbf{J}\times \mathbf{B} + \nabla\cdot\boldsymbol{\Pi}
]
Induction (with resistivity (\eta)):
[
\frac{\partial \mathbf{B}}{\partial t}=\nabla\times(\mathbf{u}\times\mathbf{B})-\nabla\times(\eta\mathbf{J}),\quad
\mathbf{J}=\frac{1}{\mu_0}\nabla\times\mathbf{B}
]

### 3.2 Energy equations with fusion source terms (two-temperature)

Ion energy:
[
\frac{\partial}{\partial t}\left(\frac{3}{2}n_i k_B T_i\right)
+\nabla\cdot\left(\frac{5}{2}n_i k_B T_i,\mathbf{u}+\mathbf{q}*i\right)
= +Q*{i,\text{fus}} - Q_{ie} - P_{i,\text{loss}}
]
Electron energy:
[
\frac{\partial}{\partial t}\left(\frac{3}{2}n_e k_B T_e\right)
+\nabla\cdot\left(\frac{5}{2}n_e k_B T_e,\mathbf{u}+\mathbf{q}*e\right)
= +Q*{e,\text{fus}} + Q_{ie} - P_{\text{rad}} - P_{e,\text{loss}}
]
Where:

* (Q_{i,\text{fus}}+Q_{e,\text{fus}} = f_{\text{dep}},P_{\text{ch}}) is the **deposited** charged-product power (see §4 for kinetic treatment).
* (P_{\text{rad}}) includes bremsstrahlung + line radiation (dominant risk for high-(Z) fuels like p–B(^\text{11})B).
* (P_{\text{loss}}) includes end losses / transport (your doc highlights **flux lifetime** as a gating metric ).

### 3.3 Compression as an MHD “driver” constraint (your pulse)

In FRC, the burn happens during rapid field ramp and adiabatic compression . A useful closure is:

* Pressure balance at high beta:
  [
  \beta \equiv \frac{2\mu_0 p}{B^2} \sim 1
  ]
  consistent with your target (\beta\approx 1) .
* Given an imposed (B(t)) (compression coil: (\sim 15\text{–}20) T ), you can treat (n(t)), (T_i(t)) as **adiabatic scalings** over the ramp (then integrate burn ODEs over the “tens of µs” burn window ).

---

## 4) Why you need hybrid kinetic (especially for aneutronic fuels)

MHD assumes near-Maxwellian thermal species. But fusion products are **born fast** (MeV) and are not Maxwellian on the burn timescale.

### 4.1 Kinetic equation for fast species (f_f(\mathbf{x},\mathbf{v},t))

A standard hybrid closure:
[
\frac{\partial f_f}{\partial t}+\mathbf{v}\cdot\nabla f_f+\frac{q_f}{m_f}\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)\cdot\nabla_{\mathbf{v}} f_f
= C_{f\leftrightarrow i,e}[f_f] + S_f
]

* (S_f) is the fusion birth source (proportional to (R_{ab})).
* (C) is Coulomb slowing down / pitch-angle scattering on bulk ions/electrons.

### 4.2 How kinetic couples back to MHD energy

The **collisional energy transfer** from fast products into bulk species becomes:
[
Q_{i,\text{fus}} = \int \frac{1}{2}m_f v^2 , C_{f\to i}, d^3v,\qquad
Q_{e,\text{fus}} = \int \frac{1}{2}m_f v^2 , C_{f\to e}, d^3v
]
This matters most for **D–(^3)He** and **p–B(^\text{11})B**, where most fusion energy is in charged products (the pathway your proposal is aiming at ).

---

## 5) Tie burn physics to inductive energy recovery (Expand → Recover)

Your doc’s recovery concept is: expanding high-pressure plasma pushes flux outward, inducing voltage in the compression coil; power electronics recapture this to recharge banks  with a target (>85%) recovery .

### 5.1 Global energy accounting (one pulse)

Define:

* Plasma thermal energy (W_{\text{th}}=\int \frac{3}{2}(n_i kT_i+n_e kT_e),dV)
* Magnetic energy (W_B=\int \frac{B^2}{2\mu_0},dV)
* Coil electrical energy in capacitor banks (W_C)

A compact pulse energy balance:
[
\frac{d}{dt}(W_{\text{th}}+W_B) ;=; P_{\text{ch,dep}} ;-; P_{\text{rad}} ;-; P_{\text{part,loss}} ;+; P_{\text{drv}}
]
where (P_{\text{drv}}) is the **electromagnetic work rate** from the driver during compression and the negative of that during expansion (generator mode).

### 5.2 Inductive recovery model (circuit coupling)

Let (\lambda(t)=N\Phi(t)) be coil flux linkage and (I(t)) the coil current:
[
V(t)= -\frac{d\lambda}{dt} - L\frac{dI}{dt} - RI
]
Recovered electrical energy over expansion:
[
E_{\text{rec}} = \eta_{\text{rec}} \int_{\text{expand}} I(t),V_{\text{useful}}(t),dt
]
The **bridge to burn equations** is via pressure-driven expansion:

* Fusion ( \Rightarrow ) higher (p=nkT) 
* higher (p) ( \Rightarrow ) stronger (p,dV) work against magnetic field
* stronger magnetic flux change (d\Phi/dt) ( \Rightarrow ) larger induced (V(t)) ( \Rightarrow ) larger (E_{\text{rec}})

### 5.3 Fuel choice impacts recoverable fraction

A clean “systems” metric for comparing fuels in this architecture:
[
\text{Recoverable fraction} ;\approx; f_{\text{ch}};f_{\text{dep}};f_{p\to \Delta \Phi};\eta_{\text{rec}}
]

* DT: (f_{\text{ch}}\sim 0.2) (most energy in neutrons) → intrinsically weaker for inductive/direct recovery.
* D–(^3)He: (f_{\text{ch}}\approx 1) but requires managing DD side neutrons .
* p–B(^\text{11})B: (f_{\text{ch}}\approx 1) but electron radiation constraints force careful (T_e) control (why two-temperature + kinetic is non-optional).

---

## 6) The minimal “burn module” you can actually integrate

Given your pulse structure and stated targets (compression to (\sim 15\text{–}20) T ; burn window tens of µs ; high beta (\sim 1) ), a practical reduced model per shot is:

1. Prescribe (B(t)) over compression/expansion; evolve (n(t),T_i(t)) via adiabatic scalings (your (n\propto B^{1.2}), (T\propto B^{0.8}) ).
2. Integrate the **species burn ODE network** (pick DT, D–(^3)He+DD, p–B).
3. Compute (P_{\text{ch}}) and deposit via a **slowing-down model** (hybrid kinetic) into (T_i,T_e).
4. Feed resulting (p(t)) into an expansion model that returns (d\Phi/dt) and thus (E_{\text{rec}}) (targeting (>85%) recovery ).
5. Check gating constraints like **flux lifetime** (\tau_\phi) vs transit+compression+burn .

If you want, I can write the full coupled ODE set explicitly for **(A) D–(^3)He with DD→T/(^3)He secondaries**, including a two-temperature electron radiation term and a simple alpha/proton slowing-down closure—so it becomes a drop-in “burn kernel” for your pulsed FRC system model.
