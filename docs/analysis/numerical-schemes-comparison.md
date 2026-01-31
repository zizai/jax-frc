# Numerical Schemes Comparison: AGATE, Athena++, and gPLUTO

## Executive Summary

This document analyzes three MHD simulation codes to inform numerical scheme choices for FRC (Field-Reversed Configuration) simulation in jax-frc.

| Feature | AGATE | Athena++ | gPLUTO |
|---------|-------|----------|--------|
| Language | Python (NumPy/Numba/CuPy) | C++ | C++ |
| Primary Use | Magnetospheric physics | Astrophysics | General MHD |
| Spatial Order | 2nd (TVD limiters) | 2nd-3rd (PLM/PPM) | 2nd-5th (MP5, WENO) |
| Time Integration | Explicit RK (SSP) | VL2, RK2-4, STS | RK2-4, IMEX, STS |
| Div(B) Control | Dedner GLM | Constrained Transport | CT, GLM, 8-wave |
| Coordinates | Cartesian only | Cart/Cyl/Sph + GR | Cart/Cyl/Sph |
| AMR | No | Yes (octree) | In development |
| GPU | Numba/CuPy/CUDA | No (CPU only) | OpenACC/OpenMP |

---

## 1. Physics Models

### 1.1 AGATE

**Implemented Models:**
- Ideal gas hydrodynamics
- Ideal MHD
- Hall MHD with ion inertial length (δi)
- CGL (Chew-Goldberger-Low) anisotropic MHD
- Anisotropic pressure MHD

**Strengths:**
- CGL model captures kinetic effects (pressure anisotropy) without full kinetic treatment
- Hall MHD for ion-scale physics
- Clean Python API for rapid prototyping

**Limitations:**
- No resistive MHD (Ohmic diffusion)
- No thermal conduction
- No viscosity

### 1.2 Athena++

**Implemented Models:**
- Ideal/isothermal hydrodynamics
- Ideal MHD
- Resistive MHD (Ohmic diffusion)
- Hall MHD
- Ambipolar diffusion
- Special/General relativistic MHD
- Radiation transport
- Cosmic ray transport

**Strengths:**
- Most comprehensive physics coverage
- Production-tested in astrophysics community
- Excellent documentation and validation

**Limitations:**
- No anisotropic pressure (CGL) model
- CPU-only (no GPU acceleration)

### 1.3 gPLUTO

**Implemented Models:**
- Hydrodynamics (HD)
- Ideal/resistive MHD
- Hall MHD
- Ambipolar diffusion
- Relativistic HD/MHD
- Resistive relativistic MHD

**Strengths:**
- Full resistive MHD with multiple diffusion mechanisms
- GPU-accelerated for large-scale simulations
- Relativistic extensions

**Limitations:**
- No anisotropic pressure model
- AMR still in development

### 1.4 Analysis for FRC

**FRC-relevant physics requirements:**
1. **Resistive MHD** - Essential for reconnection and flux decay
2. **Hall MHD** - Important for ion-scale dynamics in FRC
3. **Anisotropic pressure** - Relevant for kinetic effects in hot FRC plasmas
4. **Thermal conduction** - Important for energy transport

**Gap in jax-frc:** Currently has resistive and Hall MHD, but could benefit from:
- CGL anisotropic pressure model (from AGATE)
- Ambipolar diffusion (from Athena++/gPLUTO)

---

## 2. Spatial Discretization

### 2.1 AGATE

**Method:** Finite volume on uniform Cartesian grids

**Reconstruction:**
- MUSCL-type with TVD limiters
- Available limiters: MC, UNDTVD, low-dispersion, extremum-preserving
- Effectively 2nd-order accurate

**Stencil:** 3-5 points depending on limiter

### 2.2 Athena++

**Method:** Finite volume with Godunov-type schemes

**Reconstruction:**
- Donor Cell (1st order)
- PLM - Piecewise Linear (2nd order)
- PPM - Piecewise Parabolic (3rd order)
- Characteristic projection option

**Stencil:** 2-4 points depending on reconstruction

### 2.3 gPLUTO

**Method:** Finite volume with Godunov-type schemes

**Reconstruction:**
- FLAT (1st order)
- LINEAR/PLM (2nd order)
- PARABOLIC/PPM (3rd order)
- LimO3 (3rd order limited)
- WENO3 (3rd order WENO)
- MP5 (5th order monotonicity-preserving)
- WENOZ (WENO-Z)

**Stencil:** 2-6 points depending on reconstruction

### 2.4 Analysis for FRC

**Considerations:**
- FRC simulations often have sharp gradients (separatrix, current sheets)
- Higher-order methods reduce numerical diffusion but may oscillate near discontinuities
- TVD/WENO limiters essential for shock-capturing

**Recommendation:**
- 2nd-order PLM with MC limiter is a good baseline (robust, efficient)
- Consider WENO3 or PPM for higher accuracy in smooth regions
- MP5 may be overkill for FRC (adds complexity, marginal benefit)

---

## 3. Riemann Solvers

### 3.1 AGATE

**Available:**
- HLL (Harten-Lax-van Leer)
- HLLDiv (HLL with Dedner divergence cleaning)

**Characteristics:**
- HLL is robust but diffusive (only captures 2 waves)
- Simple implementation, good for prototyping

### 3.2 Athena++

**Available:**
- HLLE, HLLC (hydro)
- HLLD, LHLLD (MHD) - Primary choice
- Roe solver
- LLF (Local Lax-Friedrichs)

**Characteristics:**
- HLLD captures all 7 MHD waves (contact + 2 Alfvén + 4 magnetosonic)
- LHLLD reduces numerical dissipation
- Roe is most accurate but expensive

### 3.3 gPLUTO

**Available:**
- TVDLF, HLL, HLLC, HLLD, Roe, GFORCE

**Characteristics:**
- Similar to Athena++ with additional GFORCE option
- Hall MHD restricted to HLL only

### 3.4 Analysis for FRC

**FRC-specific considerations:**
- Reconnection requires accurate resolution of contact discontinuities
- HLLD significantly better than HLL for capturing current sheets
- Roe most accurate but computationally expensive

**Recommendation:**
- **HLLD** should be the default for FRC simulations
- HLL acceptable for initial testing or when Hall MHD is enabled
- Consider LHLLD for reduced numerical reconnection

**Gap in jax-frc:** Currently uses simple flux schemes. Implementing HLLD would significantly improve accuracy for reconnection studies.

---

## 4. Divergence Cleaning (∇·B = 0)

### 4.1 AGATE

**Method:** Dedner hyperbolic divergence cleaning (GLM)

**Implementation:**
- Additional scalar field ψ
- Hyperbolic transport of divergence errors at speed ch
- Exponential damping with rate cp
- Reference: Dedner et al., JCP 175 (2002)

**Pros:** Simple, works on any grid
**Cons:** Only reduces (doesn't eliminate) divergence errors

### 4.2 Athena++

**Method:** Constrained Transport (CT)

**Implementation:**
- Face-centered magnetic field (staggered grid)
- Edge-centered electric fields from Riemann solver
- CT update: dB/dt = -∇×E
- Maintains ∇·B = 0 to machine precision

**Pros:** Exact divergence-free to machine precision
**Cons:** More complex implementation, requires staggered grid

### 4.3 gPLUTO

**Methods:** Three options available

1. **EIGHT_WAVES (Powell):** Non-conservative, simple
2. **DIV_CLEANING (GLM):** Dedner method with extensions
3. **CONSTRAINED_TRANSPORT:** Multiple EMF averaging schemes (UCT_HLL, UCT_HLLD, etc.)

**Recommendation from gPLUTO:** CT is most accurate, GLM is simpler

### 4.4 Analysis for FRC

**FRC-specific considerations:**
- Magnetic topology is critical (O-point, X-points, separatrix)
- Divergence errors can corrupt topology and cause spurious reconnection
- Long-time simulations accumulate errors

**Recommendation:**
- **Constrained Transport** is strongly preferred for FRC
- GLM acceptable for short simulations or testing
- Powell 8-wave should be avoided (non-conservative)

**Gap in jax-frc:** Currently uses projection-based cleaning. Consider implementing CT for improved accuracy in long-time FRC simulations.

---

## 5. Time Integration

### 5.1 AGATE

**Explicit Methods:**
- RK11 (Forward Euler)
- RK22 (2nd order)
- RKSSP33, RKSSP43, RKSSP53 (SSP methods)

**CFL:** 0.9 (1D), 0.45 (2D), 0.32 (3D)

**Parabolic terms:** Not explicitly handled (no resistivity)

### 5.2 Athena++

**Explicit Methods:**
- VL2 (van Leer predictor-corrector) - Default
- RK1, RK2, RK3, RK4
- SSPRK(5,4)

**Super Time Stepping (STS):**
- RKL1, RKL2 (Runge-Kutta-Legendre)
- For stiff parabolic terms (diffusion)
- Reference: Meyer, Balsara & Aslam (2014)

**CFL:** 1.0 (1D), 0.5 (2D/3D) for VL2

### 5.3 gPLUTO

**Explicit Methods:**
- EULER, RK2, RK3, RK4

**Advanced Methods:**
- Super Time Stepping (STS)
- RK_CHEBYSHEV, RK_LEGENDRE
- ARK4 (Additive RK)
- IERK45 (Implicit-Explicit RK)
- SEMI_IMPLICIT

**IMEX Support:** Full IMEX capability for stiff terms

### 5.4 Analysis for FRC

**FRC-specific considerations:**
- Resistive diffusion can be stiff (small η but important)
- Hall term introduces whistler waves (very restrictive CFL)
- Thermal conduction can be stiff in hot plasmas

**Recommendation:**
- **IMEX schemes** are essential for efficient FRC simulation
- STS (Super Time Stepping) for parabolic terms
- Semi-implicit for Hall MHD to relax whistler CFL

**Current jax-frc status:** Has IMEX and semi-implicit solvers - this is good! Consider adding STS for additional efficiency with diffusion.

---

## 6. Coordinate Systems

### 6.1 AGATE

**Supported:** Cartesian only

**Limitation:** Cannot efficiently simulate cylindrical FRC geometry

### 6.2 Athena++

**Supported:**
- Cartesian (x, y, z)
- Cylindrical (R, φ, z)
- Spherical (r, θ, φ)
- GR metrics (Schwarzschild, Kerr-Schild)

**Features:** Automatic geometric source terms, polar boundary handling

### 6.3 gPLUTO

**Supported:**
- Cartesian
- Cylindrical (r, φ, z)
- Spherical (r, θ, φ)

**Features:** Metric terms (h2, h3 scale factors) for curvilinear coordinates

### 6.4 Analysis for FRC

**FRC geometry:**
- Natural coordinate system is cylindrical (r, z) with azimuthal symmetry
- 2D (r, z) simulations capture essential physics
- 3D needed for tilt instability, n≠0 modes

**Recommendation:**
- **Cylindrical coordinates** are essential for efficient FRC simulation
- 2D (r, z) axisymmetric as baseline
- 3D cylindrical for instability studies

**Current jax-frc status:** Has cylindrical coordinates - good!

---

## 7. Adaptive Mesh Refinement (AMR)

### 7.1 AGATE

**Status:** Not implemented

### 7.2 Athena++

**Status:** Full AMR support

**Features:**
- Static and adaptive refinement
- Octree-based block-structured AMR
- Flux correction at coarse-fine boundaries
- Divergence-preserving prolongation for B-field
- Load balancing

### 7.3 gPLUTO

**Status:** In development (not yet available in v0.88)

### 7.4 Analysis for FRC

**FRC-specific considerations:**
- Current sheets and separatrix need high resolution
- Bulk plasma can use coarser resolution
- AMR can provide 10-100x speedup for FRC

**Recommendation:**
- AMR is highly desirable for production FRC simulations
- Block-structured AMR (like Athena++) is well-suited
- Divergence-preserving prolongation essential for MHD

**Gap in jax-frc:** No AMR currently. This is a significant limitation for large-scale FRC simulations.

---

## 8. GPU Acceleration

### 8.1 AGATE

**Backends:**
- NumPy (baseline)
- Numba (JIT for CPU)
- Numba GPU (CUDA)
- CuPy (GPU arrays)
- CUDA RawKernel (hand-written)

**Approach:** Multi-backend with runtime selection

### 8.2 Athena++

**Status:** CPU-only (MPI + OpenMP)

**Limitation:** No GPU support

### 8.3 gPLUTO

**Backends:**
- OpenACC (NVIDIA GPUs)
- OpenMP target offload (AMD + NVIDIA)
- NCCL for multi-GPU

**Approach:** Directive-based GPU programming

### 8.4 Analysis for FRC

**Considerations:**
- FRC simulations are computationally intensive
- GPU acceleration can provide 10-50x speedup
- JAX provides automatic GPU support

**Current jax-frc status:** Uses JAX with automatic GPU support - excellent!

**Recommendation:** JAX approach is superior to directive-based (OpenACC) for:
- Automatic differentiation (useful for optimization)
- Cleaner code (no pragmas)
- Better portability (TPU support)

---

## 9. Summary Comparison Table

| Aspect | AGATE | Athena++ | gPLUTO | jax-frc |
|--------|-------|----------|--------|---------|
| **Spatial Order** | 2nd | 2nd-3rd | 2nd-5th | 2nd |
| **Riemann Solver** | HLL | HLLD | HLLD | Simple |
| **Div(B)** | GLM | CT | CT/GLM | Projection |
| **Time Integration** | Explicit RK | VL2/RK + STS | RK + IMEX | IMEX |
| **Coordinates** | Cartesian | Cart/Cyl/Sph | Cart/Cyl/Sph | Cart/Cyl |
| **AMR** | No | Yes | No | No |
| **GPU** | Multi-backend | No | OpenACC | JAX |
| **Hall MHD** | Yes | Yes | Yes | Yes |
| **Resistive MHD** | No | Yes | Yes | Yes |
| **Anisotropic P** | CGL | No | No | No |

---

## 10. Recommendations for jax-frc

### 10.1 High Priority (Significant Impact)

1. **Implement HLLD Riemann Solver**
   - Current simple flux schemes are too diffusive for reconnection
   - HLLD captures all 7 MHD waves
   - Reference: Miyoshi & Kusano, JCP 208 (2005)

2. **Implement Constrained Transport**
   - Current projection-based cleaning accumulates errors
   - CT maintains ∇·B = 0 to machine precision
   - Essential for long-time FRC simulations
   - Reference: Evans & Hawley, ApJ 332 (1988)

3. **Add Super Time Stepping (STS)**
   - Accelerates diffusion-dominated regimes
   - Complements existing IMEX solver
   - Reference: Meyer, Balsara & Aslam, JCP 257 (2014)

### 10.2 Medium Priority (Useful Enhancements)

4. **Higher-Order Reconstruction**
   - Add PPM (3rd order) option
   - Reduces numerical diffusion in smooth regions
   - Reference: Colella & Woodward, JCP 54 (1984)

5. **CGL Anisotropic Pressure Model**
   - Captures kinetic effects without full PIC
   - Relevant for hot FRC plasmas
   - Reference: Chew, Goldberger & Low, Proc. R. Soc. (1956)

6. **Ambipolar Diffusion**
   - Important for partially ionized FRC edge
   - Reference: Draine, ApJ 241 (1980)

### 10.3 Lower Priority (Future Work)

7. **Adaptive Mesh Refinement**
   - Significant implementation effort
   - High payoff for production simulations
   - Consider block-structured approach (like Athena++)

8. **WENO Reconstruction**
   - Higher-order accuracy near discontinuities
   - More complex than PPM
   - Reference: Jiang & Shu, JCP 126 (1996)

### 10.4 Current Strengths of jax-frc

- **JAX-based GPU acceleration** - Superior to directive-based approaches
- **IMEX time integration** - Essential for stiff problems
- **Semi-implicit Hall MHD** - Relaxes whistler CFL
- **Cylindrical coordinates** - Natural for FRC geometry
- **Clean Python API** - Rapid development and testing

---

## 11. Questions for Discussion

1. **Riemann solver priority:** Should HLLD be the immediate next step, or is the current scheme adequate for your use cases?

2. **Divergence cleaning:** How critical is exact ∇·B = 0 for your FRC simulations? Is GLM acceptable or is CT necessary?

3. **Anisotropic pressure:** Are you seeing kinetic effects that require CGL, or is isotropic MHD sufficient?

4. **AMR timeline:** Is AMR needed for near-term simulations, or can uniform grids suffice?

5. **Validation targets:** Which published FRC results are you comparing against? This affects which numerical schemes are most important.

---

## References

1. Dedner et al., "Hyperbolic Divergence Cleaning for the MHD Equations", JCP 175 (2002)
2. Miyoshi & Kusano, "A multi-state HLL approximate Riemann solver for ideal MHD", JCP 208 (2005)
3. Evans & Hawley, "Simulation of Magnetohydrodynamic Flows: A Constrained Transport Method", ApJ 332 (1988)
4. Meyer, Balsara & Aslam, "A stabilized Runge-Kutta-Legendre method for explicit super-time-stepping", JCP 257 (2014)
5. Colella & Woodward, "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations", JCP 54 (1984)
6. Stone et al., "Athena: A New Code for Astrophysical MHD", ApJS 178 (2008)
7. Mignone et al., "PLUTO: A Numerical Code for Computational Astrophysics", ApJS 170 (2007)
