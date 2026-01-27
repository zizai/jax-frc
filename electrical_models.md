### Key Points

* Research suggests that circuit-plasma coupling in FRCs primarily occurs through mutual inductance between external coils and plasma currents, facilitating inductive energy transfer during formation, though efficiency can be limited by plasma resistance and geometry.
* Charging and acceleration models typically involve capacitor banks discharging into theta-pinch or RMF coils to generate poloidal fields, with plasma acceleration driven by Lorentz forces and magnetic gradients, enabling translation at Alfvénic speeds.
* Energy recovery via magnetic induction leverages FRC expansion in low-field chambers, compressing bias fields to induce liner currents, with evidence indicating potential efficiencies up to 40% under optimized separatrix ratios, though practical implementations face challenges in material limits and timing.
* For FRC fusion, these models often assume axisymmetric geometries and adiabatic processes, with critiques highlighting the need for better integration of kinetic effects to avoid overestimating coupling losses.

### Derivations Overview

Circuit-plasma coupling equations derive from lumped circuit approximations, incorporating mutual inductances for inductive transfer. Charging/acceleration models build from Faraday's law and Lorentz force, while energy recovery equations stem from energy balance during adiabatic expansion, conserving flux and converting plasma energy to electrical via induced EMFs.

### Numerical Methods Summary

Simulations use equivalent circuit solvers like PSpice for transient analysis, coupled with MHD codes (e.g., NIMROD) for plasma dynamics; implicit time-stepping handles stiffness from high-frequency RMF, with finite element methods for field distributions.

### FRC Adaptations

Practical assumptions include quasi-neutrality, negligible electron inertia, and uniform resistivity; adaptations for fusion reactors incorporate anomalous resistivity for reconnection and reduced dimensions (2D axisymmetric) to manage computational costs, focusing on high-β equilibria.

### Analysis and Comparison

Circuit models efficiently predict coupling but overlook kinetic details; acceleration schemes excel in pulsed systems for rapid translation, while energy recovery enhances overall efficiency but requires precise timing. Critiques note idealizations in inductance assumptions versus real non-uniform plasmas, with hybrids offering better fidelity at higher cost.

---

This deep-dive into electrical models for Field-Reversed Configurations (FRCs) in fusion reactors focuses on deriving equations for circuit-plasma coupling, charging/acceleration, and energy recovery through magnetic induction. Drawing from plasma physics fundamentals and engineering approximations, the derivations start from Maxwell's equations and circuit theory, applying simplifications suited to FRC's high-β, compact toroid nature. Step-by-step processes are provided, with analytic examples where possible, and validations extracted from benchmarks like those in RMF thrusters and staged compression reactors. Tables summarize key parameters and comparisons for clarity. The models emphasize inductive coupling via mutual inductances, Lorentz-driven dynamics, and flux-conserving expansions, relevant to devices like TAE's C-2W or Princeton's PFRC.

#### Step-by-Step Derivation of Circuit-Plasma Coupling Equations

Circuit-plasma coupling in FRCs models the interaction between external electrical circuits (e.g., capacitor banks, coils) and plasma currents as coupled inductors, enabling energy transfer for formation and sustainment. This is crucial in theta-pinch or Rotating Magnetic Field (RMF) systems, where external fields induce azimuthal plasma currents.

1. **Starting from Maxwell's Equations**: Begin with Ampère's law with Maxwell correction (neglecting displacement for low frequencies): (\nabla \times \vec{B} = \mu_0 \vec{j}), and Faraday's law: (\nabla \times \vec{E} = -\partial \vec{B}/\partial t).

2. **Lumped Circuit Approximation**: Treat coils and plasma as current loops. For a coil with current (I_c) and plasma with current (I_p), mutual inductance (M) links fluxes: Induced EMF in plasma (\mathcal{E}_p = -M dI_c/dt), and vice versa (\mathcal{E}_c = -M dI_p/dt).

3. **Generalized Circuit Equations**: For an RMF antenna system (x and y coils orthogonal), include self-inductances (L_c, L_p), resistances (R_c, R_p), and capacitance (C):
   [
   L_c \frac{dI_{c,x}}{dt} + R_c I_{c,x} + \frac{1}{C} \int I_{c,x} dt + M \frac{dI_p}{dt} = V_0 \sin(\omega t)
   ]
   [
   L_p \frac{dI_p}{dt} + R_p I_p + M \frac{dI_{c,x}}{dt} = 0
   ]
   where (V_0) is drive voltage, (\omega) RMF frequency.

4. **Mutual Inductance Derivation**: From Biot-Savart, for axisymmetric FRC:
   [
   M = \pi \int_{V_p} g_z(\vec{r}) \gamma_c(\vec{r}) \sin\theta , dV_p
   ]
   where (g_z) is axial plasma current density, (\gamma_c = B_c / I_c) geometric factor (coil field per current), integrated over plasma volume (V_p).

5. **Incorporating Plasma Dynamics**: Add gyrator terms for Lorentz coupling (Hall effects):
   [
   \Gamma_c = \pi \int_{V_p} g_z g_\theta n_e \gamma_c \cos\theta , dV_p
   ]
   where (g_\theta) azimuthal current density, (n_e) electron density. This yields effective resistance from drifts.

6. **Effective Mutual Inductance for Decaying Coupling**: In translating FRCs:
   [
   M_{\text{eff}} = M_0 e^{-z/z_0}
   ]
   where (z) is axial position, (z_0) stroke length (coupling decay scale).

Analytic Example: For steady-state RMF, plasma current (I_p \approx (M \omega I_c)/R_p), yielding coupling efficiency (\eta \approx (M^2 \omega^2)/(R_p^2 + M^2 \omega^2)); benchmarks show ~53% in prototypes with (I_p \sim 1.9) kA.

#### Step-by-Step Derivation of Charging/Acceleration Equations

Charging involves capacitor banks building voltage for coil discharge, while acceleration uses magnetic gradients to propel FRC plasmoids, often in conical geometries for fusion compression.

1. **Charging Circuit**: From Kirchhoff's laws for capacitor-coil system:
   [
   V_{\text{bank}} = \frac{Q}{C} + L \frac{dI}{dt} + R I + M \frac{dI_p}{dt}
   ]
   where (Q = \int I dt), solved as damped oscillator (\omega = 1/\sqrt{LC}).

2. **Plasma Induction During Charging**: Induced plasma field from Ohm's law (\vec{E} + \vec{v} \times \vec{B} = \eta \vec{j}), yielding current ramp (dI_p/dt \approx (M/L_p) dI_c/dt).

3. **Acceleration via Lorentz Force**: Plasma slug motion from Newton's law:
   [
   m_s \frac{d^2 z}{dt^2} = \int \vec{j}*p \times \vec{B}*{\text{ext}} , dV_p
   ]
   Approximate as:
   [
   m_s \ddot{z} = \alpha I_c^2 e^{-z/z_0}
   ]
   where (\alpha = \pi \int g_\theta B_{r,\text{ext}} dV_p) acceleration coefficient.

4. **Coupled Charging-Acceleration**: Full system:
   [
   \left( L_c - M_{\text{eff}} \right) \ddot{I}*{c,x} + \left( R_c - 2 \dot{M}*{\text{eff}} \right) \dot{I}*{c,x} + \left( \frac{1}{C} + \ddot{M}*{\text{eff}} \right) I_{c,x} = \omega V_0 \sin(\omega t)
   ]
   [
   m_s \ddot{z} = \alpha (\dot{I}*{c,x} I*{c,x} - R_s M_{\text{eff}}) \tanh(t/\tau) e^{-z/z_0}
   ]
   with (\tau) ionization time, (R_s) plasma resistance.

Analytic Example: Super-Alfvénic acceleration (v_z \sim 0.4 v_A) in merging, validated against experiments with translation in ~5-7 Alfvén times.

#### Step-by-Step Derivation of Energy Recovery Equations via Magnetic Induction

Energy recovery captures FRC expansion energy by compressing bias fields in a low-field chamber, inducing liner currents.

1. **Energy Balance**: Initial FRC energy partitions:
   [
   E_i^{\text{FRC}} = E_f^{\text{FRC}} + W_B
   ]
   where (W_B) work on bias field.

2. **Magnetic Work Derivation**: From field energy density:
   [
   W_B = \int \frac{B^2 - B_0^2}{2\mu_0} dV = \frac{(1 - x_s^2) x_s^2 B_e^2}{2\mu_0} V_{\text{liner}}
   ]
   with (B_0 = (1 - x_s^2) B_e), (x_s = r_s / r_{\text{liner}}) separatrix ratio.

3. **Final FRC Energy**: Adiabatic assumption ((p V^\gamma = const)):
   [
   E_f^{\text{FRC}} = \frac{3}{2} \frac{B_e^2}{2\mu_0} x_s^2 V_{\text{liner}} = \frac{3}{2} \frac{1}{1 - x_s^2} W_B
   ]

4. **Induced Liner Current**: Flux conservation (\phi = \pi r^2 B = const):
   [
   I_{\text{liner}} = \frac{l_{\text{liner}} (B_e - B_0)}{\mu_0} = \frac{l_{\text{liner}} x_s^2 B_e}{\mu_0}
   ]
   Energy extracted (W_B = \frac{1}{2} \phi_0 I_{\text{liner}}).

5. **Efficiency**:
   [
   \eta = \frac{W_B}{E_i^{\text{FRC}}} = 1 - \frac{x_s^2 (1 - x_s^2)}{2.5 (1 - x_s^2) - x_s^2}
   ]
   Max ~40% at (x_s \approx 0.33).

Analytic Example: For 2 MJ FRC, (B_0 \sim 2) T, expansion yields ~0.8 MJ recovered, matching simulations with large radius ratios (~14:1).

#### Validation Configurations, Parameters, and Target Results from Benchmarks

Benchmarks from RMF thrusters and staged compression reactors validate models.

| Model Aspect            | Benchmark Source                    | Configuration                                  | Key Parameters                                                                                       | Target Results/Validation                                                                              |
| ----------------------- | ----------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Circuit-Plasma Coupling | RMF Thruster (AIAA 2021)            | RMF antennae driving FRC formation             | ( \omega = 2\pi \times 10^6 ) rad/s, ( n_e = 10^{19} ) m^{-3}, ( M_0 = 10^{-6} ) H, ( z_0 = 0.15 ) m | Coupling efficiency ~53%, plasma current 1.9 kA, axial B reversal 155 G; matches exp. within 10%.      |
| Charging/Acceleration   | Supersonic Merging (POP 28, 022101) | Capacitor discharge into coils for translation | ( V_z = 0.1-0.2 v_A ), Lundquist S=4600, separation (\Delta Z = 180 d_i)                             | Acceleration to 0.4 v_A, merged FRC density/temperature agree with TAE C-2 exp. (~10-20%).             |
| Energy Recovery         | Staged Compression (NF 62, 2022)    | FRC expansion in ERC with liner                | ( x_s = 0.33 ), ( B_0 = 2 ) T, ( V_{\text{liner}} = \pi (0.5)^2 \times 1 ) m^3, initial E=2 MJ       | Recovery ~38% (0.76 MJ), final B_e ~6 T; aligns with adiabatic models, efficiency within 5% of theory. |

#### Analysis, Comparison, and Critiques

**Analysis**: Circuit coupling enables efficient FRC formation (~50-60% energy transfer) via inductive RMF, with acceleration achieving super-Alfvénic speeds for compression heating to keV temperatures. Energy recovery converts post-burn plasma energy to electricity, enhancing reactor Q (gain) by recycling ~40% of output.

**Comparison**:

* Coupling vs. Acceleration: Coupling focuses on inductive efficiency (high in RMF, >50%), while acceleration adds dynamic losses (exponential decay with z).
* Recovery vs. Others: Unique in pulsed FRCs, outperforming steady-state in efficiency but requiring precise ejection timing.

| Aspect          | Circuit Coupling      | Charging/Acceleration   | Energy Recovery      |
| --------------- | --------------------- | ----------------------- | -------------------- |
| Physics         | Mutual inductance     | Lorentz force           | Flux compression     |
| Efficiency      | 50-60%                | Thrust ~ ω B            | Up to 40%            |
| FRC Suitability | Formation/sustainment | Translation/compression | Post-burn extraction |
| Limitations     | Plasma resistance     | Geometric decay         | Material stresses    |

**Critiques**: Models assume uniform densities, underestimating kinetic losses; efficiency drops in non-ideal plasmas. Recovery relies on HTSC tech, unproven at scale; full kinetic simulations needed for microinstabilities. Progress requires integrated MHD-circuit codes for reactor design.

### Key Citations

* Equivalent Circuit Model for a Rotating Magnetic Field Thruster: [https://pepl.engin.umich.edu/pdf/2021_AIAA_PE_Woods.pdf](https://pepl.engin.umich.edu/pdf/2021_AIAA_PE_Woods.pdf)
* A compact fusion reactor based on the staged compression of an FRC: [https://iopscience.iop.org/article/10.1088/1741-4326/ae034d/pdf](https://iopscience.iop.org/article/10.1088/1741-4326/ae034d/pdf)
* Circuit Simulations for the RMFO/FRC Antenna System: [https://w3.pppl.gov/ppst/docs/sapan2002.pdf](https://w3.pppl.gov/ppst/docs/sapan2002.pdf)
* The Fork in the Road to Electric Power From Fusion Helion Energy: [https://lynceans.org/wp-content/uploads/2021/02/Helion-Energy_US-converted.pdf](https://lynceans.org/wp-content/uploads/2021/02/Helion-Energy_US-converted.pdf)
* Equivalent Circuit Model for a Rotating Magnetic Field Thruster: [https://pepl.engin.umich.edu/pdf/2021_AIAA_PE_Woods.pdf](https://pepl.engin.umich.edu/pdf/2021_AIAA_PE_Woods.pdf)
* Defense Intelligence Reference Document Aneutronic Fusion Propulsion: [https://www.dia.mil/FOIA/FOIA-Electronic-Reading-Room/FileId/237644](https://www.dia.mil/FOIA/FOIA-Electronic-Reading-Room/FileId/237644)
* Research Article Formation of Field Reversed Configuration (FRC) on the Yingguang-I device: [https://www.sciencedirect.com/science/article/pii/S2468080X17300857](https://www.sciencedirect.com/science/article/pii/S2468080X17300857)
* Design and experimental study of a field-reversed configuration plasma thruster prototype: [https://pubs-en.cstam.org.cn/data/article/pst/preview/pdf/PST-2024-0276.pdf](https://pubs-en.cstam.org.cn/data/article/pst/preview/pdf/PST-2024-0276.pdf)
* Pulsed Inductive Plasma Acceleration: Performance Optimization Criteria: [https://ntrs.nasa.gov/api/citations/20140012846/downloads/20140012846.pdf](https://ntrs.nasa.gov/api/citations/20140012846/downloads/20140012846.pdf)
* Numerical Investigation of the Formation and Translation of a Field Reversed Configuration with Neutrals and External Circuits Effects: [https://digital.lib.washington.edu/researchworks/items/cc377c76-8fd5-4933-bc80-27a392f0162d](https://digital.lib.washington.edu/researchworks/items/cc377c76-8fd5-4933-bc80-27a392f0162d)
* (PDF) Magnetic feedback control of a Field Reversed Configuration: [https://www.researchgate.net/publication/398290468_Magnetic_feedback_control_of_a_Field_Reversed_Configuration](https://www.researchgate.net/publication/398290468_Magnetic_feedback_control_of_a_Field_Reversed_Configuration)
* Progress and issues with pulsed magnetic fusion | Physics of Plasmas: [https://pubs.aip.org/aip/pop/article/32/2/022507/3336518/Progress-and-issues-with-pulsed-magnetic-fusion](https://pubs.aip.org/aip/pop/article/32/2/022507/3336518/Progress-and-issues-with-pulsed-magnetic-fusion)
* The Princeton Field-Reversed Configuration for Compact Nuclear Fusion Power Plants: [https://w3.pppl.gov/ppst/docs/galea2023jfe.pdf](https://w3.pppl.gov/ppst/docs/galea2023jfe.pdf)
* Fundamental Scaling of Adiabatic Compression of Field Reversed Configuration Thermonuclear Fusion Plasmas: [https://link.springer.com/article/10.1007/s10894-023-00367-7](https://link.springer.com/article/10.1007/s10894-023-00367-7)
* Enhanced plasma performance in C-2W advanced beam-driven field-reversed configuration experiments: [https://iopscience.iop.org/article/10.1088/1741-4326/ad4536](https://iopscience.iop.org/article/10.1088/1741-4326/ad4536)
* Hybrid simulations of FRC merging and compression: [https://arxiv.org/html/2501.03425v1](https://arxiv.org/html/2501.03425v1)
* Fundamental Scaling of Adiabatic Compression of Field Reversed Configuration Thermonuclear Fusion Plasmas: [https://link.springer.com/article/10.1007/s10894-023-00367-7](https://link.springer.com/article/10.1007/s10894-023-00367-7)
* US6850011B2 - Controlled fusion in a field reversed configuration and direct energy conversion: [https://patents.google.com/patent/US6850011B2/en](https://patents.google.com/patent/US6850011B2/en)
* Research Article Formation of Field Reversed Configuration (FRC) on the Yingguang-I device: [https://www.sciencedirect.com/science/article/pii/S2468080X17300857](https://www.sciencedirect.com/science/article/pii/S2468080X17300857)
* The Princeton Field-Reversed Configuration for Compact Nuclear Fusion Power Plants: [https://www.researchgate.net/publication/367638215_The_Princeton_Field-Reversed_Configuration_for_Compact_Nuclear_Fusion_Power_Plants](https://www.researchgate.net/publication/367638215_The_Princeton_Field-Reversed_Configuration_for_Compact_Nuclear_Fusion_Power_Plants)
