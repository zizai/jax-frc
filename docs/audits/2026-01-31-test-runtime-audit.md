# Test Runtime Audit (Not-Slow Suite)

**Date:** 2026-01-31  
**Focus:** Solver-heavy unit tests  
**Command:** `./.venv/bin/python -m pytest tests/test_imex_diffusion.py tests/test_imex_validation.py tests/test_hlld_solver.py tests/test_divergence_cleaning_3d.py tests/test_energy_integration.py tests/test_equilibrium_3d.py tests/test_extended_mhd_3d.py tests/test_hybrid_kinetic_3d.py tests/test_finite_volume_mhd.py --durations=20 -q`  
**Environment:** local venv in `.worktrees/test-optimization`

## Slowest Tests (Top 20)

1. **9.27s** — `tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_1d_diffusion_decay_rate`  
   Likely heavy: IMEX diffusion solve loop, diffusion RHS, time stepping.
2. **7.28s** — `tests/test_imex_diffusion.py::test_gaussian_diffusion_converges`  
   Likely heavy: IMEX diffusion integration, grid size + timestep count.
3. **6.61s** — `tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_decay_rate_scales_with_resistivity`  
   Likely heavy: repeated diffusion solves across multiple resistivity values.
4. **6.57s** — `tests/test_imex_diffusion.py::test_imex_large_timestep_stable`  
   Likely heavy: large timestep integration, possible stabilization iterations.
5. **4.26s** — `tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_diffusion_preserves_pattern_shape`  
   Likely heavy: diffusion integration with shape comparison.
6. **3.52s** — `tests/test_energy_integration.py::TestHeatConductionAnalytical::test_1d_heat_conduction_diffusion`  
   Likely heavy: thermal diffusion integration loop.
7. **3.36s** — `tests/test_finite_volume_mhd.py::test_finite_volume_mhd_uniform_state_zero_rhs`  
   Likely heavy: finite volume flux evaluation + divergence.
8. **3.03s** — `tests/test_energy_integration.py::TestEndToEndTemperatureEvolution::test_simulation_runs_without_error`  
   Likely heavy: end-to-end time stepping.
9. **2.90s** — `tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_uniform_field_is_stationary`  
   Likely heavy: IMEX diffusion integration.
10. **2.66s** — `tests/test_hybrid_kinetic_3d.py::TestHybridParticles3D::test_deposit_current_shape_3d`  
    Likely heavy: particle deposition and grid interpolation.
11. **2.23s** — `tests/test_equilibrium_3d.py::TestForceBalanceSolver::test_solve_with_harris_initial`  
    Likely heavy: force-balance solver iterations.
12. **1.70s** — `tests/test_hybrid_kinetic_3d.py::TestHybridKinetic3D::test_compute_rhs_asymmetric_grid`  
    Likely heavy: 3D RHS evaluation.
13. **1.68s** — `tests/test_hlld_solver.py::test_hlld_flux_matches_physical_flux_for_equal_states`  
    Likely heavy: HLLE/HLLD flux computation.
14. **1.59s** — `tests/test_extended_mhd_3d.py::TestExtendedMHD3D::test_hall_term_present`  
    Likely heavy: extended MHD RHS path.
15. **1.50s** — `tests/test_divergence_cleaning_3d.py::TestCleanDivergence::test_already_divergence_free`  
    Likely heavy: Poisson solve / divergence cleaning.
16. **1.44s** — `tests/test_hybrid_kinetic_3d.py::TestHybridParticles3D::test_push_particles_3d`  
    Likely heavy: particle push in 3D.
17. **1.42s** — `tests/test_equilibrium_3d.py::TestForceBalanceSolver::test_solve_returns_state`  
    Likely heavy: force-balance solver iterations.
18. **1.30s** — `tests/test_equilibrium_3d.py::TestEquilibriumPhysics::test_divergence_free_flux_rope`  
    Likely heavy: flux rope setup + divergence checks.
19. **1.25s** — `tests/test_energy_integration.py::TestThermalDiffusionConservation::test_diffusion_conserves_total_thermal_energy`  
    Likely heavy: diffusion integration.
20. **1.20s** — `tests/test_hybrid_kinetic_3d.py::TestHybridKinetic3D::test_compute_rhs_shape`  
    Likely heavy: 3D RHS evaluation.

## Next Actions (Planned Optimizations)

- Reduce grid sizes and/or time steps in IMEX diffusion tests.
- Collapse resistivity sweeps to 1–2 representative values.
- Shorten end-to-end energy integration runs while preserving invariants.
- Reduce particle counts and grid sizes in hybrid kinetic tests.
- Reduce equilibrium solver iterations or loosen tolerances for unit tests.

## Model-Focused Tests (Additional Audit)

**Command:**  
`./.venv/bin/python -m pytest tests/test_resistive_mhd_3d.py tests/test_extended_mhd_3d.py tests/test_resistive_mhd_split.py tests/test_mhd_rhs_consistency.py tests/test_mhd_solver_consistency.py tests/test_mhd_mms_convergence.py tests/test_integration_3d.py tests/test_simulation_integration.py tests/test_burning_plasma.py tests/test_burning_plasma_circuits.py tests/test_neutral_fluid.py tests/test_neutral_fluid_3d.py tests/test_coupled_model.py --durations=20 -q`

**Slowest 20 durations:**

1. **8.74s** — `tests/test_mhd_mms_convergence.py::test_extended_mhd_mms_hall_ep_convergence`  
   Likely heavy: MMS convergence grid/step sweeps.
2. **4.93s** — `tests/test_simulation_integration.py::TestResistiveMHD::test_flux_diffusion`  
   Likely heavy: time-stepping integration.
3. **3.86s** — `tests/test_mhd_mms_convergence.py::test_resistive_mhd_mms_convergence`  
   Likely heavy: MMS convergence grid/step sweeps.
4. **3.77s** — `tests/test_mhd_solver_consistency.py::test_imex_tiny_dt_rhs_consistency`  
   Likely heavy: IMEX consistency checks with tiny dt.
5. **3.71s** — `tests/test_simulation_integration.py::TestEnergyConservation::test_magnetic_energy_decay`  
   Likely heavy: multi-step energy evolution.
6. **2.90s** — `tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_from_config_with_recipe`  
   Likely heavy: full config-driven simulation setup + steps.
7. **2.66s** — `tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_run_steps`  
   Likely heavy: integration steps.
8. **2.53s** — `tests/test_integration_3d.py::TestResistiveMHDIntegration::test_resistive_mhd_time_stepping`  
   Likely heavy: 3D time stepping.
9. **2.21s** — `tests/test_integration_3d.py::TestExtendedMHDIntegration::test_hall_term_affects_evolution`  
   Likely heavy: 3D hall-term integration.
10. **2.00s** — `tests/test_coupled_model.py::test_coupled_model_explicit_rhs`  
    Likely heavy: coupled RHS evaluation.
11. **1.93s** — `tests/test_simulation_integration.py::TestHybridKinetic::test_boris_pusher`  
    Likely heavy: particle push integration.
12. **1.80s** — `tests/test_mhd_rhs_consistency.py::test_resistive_mhd_rk4_tiny_dt_rhs_consistency`  
    Likely heavy: RK4 RHS consistency checks.
13. **1.70s** — `tests/test_extended_mhd_3d.py::TestExtendedMHD3D::test_hall_term_present`  
    Likely heavy: extended MHD RHS path.
14. **1.54s** — `tests/test_mhd_mms_convergence.py::test_extended_mhd_mms_convergence`  
    Likely heavy: MMS convergence grid/step sweeps.
15. **1.52s** — `tests/test_neutral_fluid_3d.py::TestNeutralFluid3D::test_flux_divergence_asymmetric_shape`  
    Likely heavy: 3D flux divergence.
16. **1.49s** — `tests/test_mhd_rhs_consistency.py::test_extended_mhd_rk4_tiny_dt_rhs_consistency`  
    Likely heavy: RK4 RHS consistency checks.
17. **1.42s** — `tests/test_resistive_mhd_split.py::test_compute_rhs_nonzero_for_nonuniform_b`  
    Likely heavy: RHS eval.
18. **1.30s** — `tests/test_mhd_solver_consistency.py::test_imex_residual_scales_with_dt`  
    Likely heavy: IMEX residual checks.
19. **1.25s** — `tests/test_simulation_integration.py::TestExtendedMHD::test_hall_term_present`  
    Likely heavy: integration setup + RHS.
20. **1.24s** — `tests/test_resistive_mhd_3d.py::TestResistiveMHD3D::test_compute_rhs_shape`  
   Likely heavy: RHS evaluation.

## Diffusion 3D (Pseudo-2D Update)

**Command:** `timeout 30s ./.venv/bin/python -m pytest tests/test_diffusion_3d.py -v`  
**Result:** 4 passed in **6.58s**

**Change:** collapsed grid to 16x16x1 with 2D analytic scaling.

## IMEX Tests (Reduced CG Iterations)

**Command:** `./.venv/bin/python -m pytest tests/test_imex_diffusion.py tests/test_imex_validation.py -v`  
**Result:** 6 passed in **24.54s**

**Changes:** smaller grids/steps and `cg_max_iter=10` with relaxed tolerances.
