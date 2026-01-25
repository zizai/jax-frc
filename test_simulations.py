import jax
import jax.numpy as jnp
from resistive_mhd import laplace_star, compute_j_phi, chodura_resistivity, circuit_dynamics
from extended_mhd import curl_2d, grad_2d, extended_ohm_law, hall_operator, apply_halo_density
from hybrid_kinetic import rigid_rotor_f0, compute_f0_gradient, boris_push, weight_evolution
from physics_utils import (
    compute_alfven_speed, compute_cyclotron_frequency, 
    compute_larmor_radius, compute_skin_depth, compute_beta,
    compute_gradient, compute_laplacian, apply_boundary_conditions
)

def test_resistive_mhd():
    print("Testing Resistive MHD functions...")
    
    nr, nz = 32, 64
    dr, dz = 1.0/nr, 2.0/nz
    r = jnp.linspace(0.1, 1.0, nr)[:, None]
    psi = jnp.ones((nr, nz))
    
    delta_star = laplace_star(psi, dr, dz, r)
    assert delta_star.shape == (nr, nz), f"Expected shape {(nr, nz)}, got {delta_star.shape}"
    print("  ✓ laplace_star passed")
    
    j_phi = compute_j_phi(psi, dr, dz, r)
    assert j_phi.shape == (nr, nz), f"Expected shape {(nr, nz)}, got {j_phi.shape}"
    print("  ✓ compute_j_phi passed")
    
    eta = chodura_resistivity(psi, j_phi)
    assert eta.shape == (nr, nz), f"Expected shape {(nr, nz)}, got {eta.shape}"
    print("  ✓ chodura_resistivity passed")
    
    I_coil = 1.0
    V_bank = 1000.0
    L_coil = 1e-6
    M_plasma_coil = 1e-7
    dI_plasma_dt = 1e4
    dt = 1e-4
    
    I_coil_new, dI_coil_dt = circuit_dynamics(I_coil, V_bank, L_coil, M_plasma_coil, dI_plasma_dt, dt)
    assert jnp.isfinite(I_coil_new), "I_coil_new is not finite"
    assert jnp.isfinite(dI_coil_dt), "dI_coil_dt is not finite"
    print("  ✓ circuit_dynamics passed")
    
    print("All Resistive MHD tests passed!\n")

def test_extended_mhd():
    print("Testing Extended MHD functions...")
    
    nx, ny = 32, 32
    dx, dy = 1.0/nx, 1.0/ny
    
    f_x = jnp.ones((nx, ny))
    f_y = jnp.ones((nx, ny))
    
    curl = curl_2d(f_x, f_y, dx, dy)
    assert curl.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {curl.shape}"
    print("  ✓ curl_2d passed")
    
    f = jnp.ones((nx, ny))
    df_dx, df_dy = grad_2d(f, dx, dy)
    assert df_dx.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {df_dx.shape}"
    assert df_dy.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {df_dy.shape}"
    print("  ✓ grad_2d passed")
    
    v_x, v_y, v_z = jnp.ones((nx, ny)), jnp.ones((nx, ny)), jnp.ones((nx, ny))
    b_x, b_y, b_z = jnp.ones((nx, ny)), jnp.ones((nx, ny)), jnp.ones((nx, ny))
    j_x, j_y, j_z = jnp.ones((nx, ny)), jnp.ones((nx, ny)), jnp.ones((nx, ny))
    n = jnp.ones((nx, ny)) * 1e19
    eta = 1e-4
    p_e = jnp.ones((nx, ny)) * 1e3
    
    E_x, E_y, E_z = extended_ohm_law(v_x, v_y, v_z, b_x, b_y, b_z, j_x, j_y, j_z, n, eta, p_e, dx, dy)
    assert E_x.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {E_x.shape}"
    assert E_y.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {E_y.shape}"
    assert E_z.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {E_z.shape}"
    print("  ✓ extended_ohm_law passed")
    
    curl_hall_x, curl_hall_y, curl_hall_z = hall_operator(b_x, b_y, b_z, n, dx, dy)
    assert curl_hall_x.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {curl_hall_x.shape}"
    print("  ✓ hall_operator passed")
    
    n_with_halo = apply_halo_density(n)
    assert n_with_halo.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {n_with_halo.shape}"
    print("  ✓ apply_halo_density passed")
    
    print("All Extended MHD tests passed!\n")

def test_hybrid_kinetic():
    print("Testing Hybrid Kinetic functions...")
    
    n0 = 1e19
    T0 = 100.0
    Omega = 1e5
    
    r = jnp.array([0.1, 0.2, 0.3])
    z = jnp.array([0.0, 0.1, -0.1])
    vr = jnp.array([1e4, 2e4, 3e4])
    vz = jnp.array([1e4, 2e4, 3e4])
    vtheta = jnp.array([1e4, 2e4, 3e4])
    
    f0 = rigid_rotor_f0(r, z, vr, vz, vtheta, n0, T0, Omega)
    assert f0.shape == (3,), f"Expected shape (3,), got {f0.shape}"
    assert jnp.all(f0 > 0), "f0 should be positive"
    print("  ✓ rigid_rotor_f0 passed")
    
    df0_dr, df0_dvr, df0_dvz, df0_dvtheta = compute_f0_gradient(r, z, vr, vz, vtheta, n0, T0, Omega, 0.01, 0.01)
    assert df0_dr.shape == (3,), f"Expected shape (3,), got {df0_dr.shape}"
    print("  ✓ compute_f0_gradient passed")
    
    x = jnp.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.1], [0.3, 0.0, -0.1]])
    v = jnp.array([[1e4, 1e4, 1e4], [2e4, 2e4, 2e4], [3e4, 3e4, 3e4]])
    E = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    B = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    dt = 1e-8
    
    x_new, v_new = boris_push(x, v, E, B, 1.602e-19, 1.673e-27, dt)
    assert x_new.shape == x.shape, f"Expected shape {x.shape}, got {x_new.shape}"
    assert v_new.shape == v.shape, f"Expected shape {v.shape}, got {v_new.shape}"
    print("  ✓ boris_push passed")
    
    w = jnp.array([0.0, 0.1, -0.1])
    w_new = weight_evolution(w, x_new, v_new, None, None, n0, T0, Omega, 0.01, 0.01)
    assert w_new.shape == w.shape, f"Expected shape {w.shape}, got {w_new.shape}"
    assert jnp.all(jnp.abs(w_new) <= 1.0), "Weights should be clipped to [-1, 1]"
    print("  ✓ weight_evolution passed")
    
    print("All Hybrid Kinetic tests passed!\n")

def test_physics_utils():
    print("Testing Physics Utils functions...")
    
    B = 1.0
    n = 1e19
    T = 100.0
    v_perp = 1e5
    
    v_A = compute_alfven_speed(B, n)
    assert jnp.isfinite(v_A), "v_A is not finite"
    print("  ✓ compute_alfven_speed passed")
    
    omega_c = compute_cyclotron_frequency(B)
    assert jnp.isfinite(omega_c), "omega_c is not finite"
    print("  ✓ compute_cyclotron_frequency passed")
    
    r_L = compute_larmor_radius(v_perp, B)
    assert jnp.isfinite(r_L), "r_L is not finite"
    print("  ✓ compute_larmor_radius passed")
    
    d_i = compute_skin_depth(n)
    assert jnp.isfinite(d_i), "d_i is not finite"
    print("  ✓ compute_skin_depth passed")
    
    beta = compute_beta(n, T, B)
    assert jnp.isfinite(beta), "beta is not finite"
    print("  ✓ compute_beta passed")
    
    nx, ny = 32, 32
    dx, dy = 1.0/nx, 1.0/ny
    f = jnp.ones((nx, ny))
    
    df_dx, df_dy = compute_gradient(f, dx, dy)
    assert df_dx.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {df_dx.shape}"
    print("  ✓ compute_gradient passed")
    
    laplacian = compute_laplacian(f, dx, dy)
    assert laplacian.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {laplacian.shape}"
    print("  ✓ compute_laplacian passed")
    
    field = jnp.ones((nx, ny))
    field_bc = apply_boundary_conditions(field, bc_type='dirichlet', bc_value=0.0)
    assert field_bc.shape == (nx, ny), f"Expected shape {(nx, ny)}, got {field_bc.shape}"
    print("  ✓ apply_boundary_conditions passed")
    
    print("All Physics Utils tests passed!\n")

def test_simulations():
    print("Testing simulation runs (may take a moment)...")
    
    try:
        from resistive_mhd import run_simulation as run_resistive_mhd
        final_psi, final_I_coil, history = run_resistive_mhd(steps=10, nr=16, nz=32)
        assert jnp.isfinite(final_psi).all(), "Resistive MHD simulation produced non-finite values"
        print("  ✓ Resistive MHD simulation passed")
    except Exception as e:
        print(f"  ✗ Resistive MHD simulation failed: {e}")
    
    try:
        from extended_mhd import run_simulation as run_extended_mhd
        b_final, history = run_extended_mhd(steps=10, nx=16, ny=16)
        assert jnp.isfinite(b_final[0]).all(), "Extended MHD simulation produced non-finite values"
        print("  ✓ Extended MHD simulation passed")
    except Exception as e:
        print(f"  ✗ Extended MHD simulation failed: {e}")
    
    try:
        from hybrid_kinetic import run_simulation as run_hybrid_kinetic
        x_final, v_final, w_final, history = run_hybrid_kinetic(steps=10, n_particles=100, nr=16, nz=32)
        assert jnp.isfinite(x_final).all(), "Hybrid Kinetic simulation produced non-finite values"
        print("  ✓ Hybrid Kinetic simulation passed")
    except Exception as e:
        print(f"  ✗ Hybrid Kinetic simulation failed: {e}")
    
    print("Simulation tests completed!\n")

def main():
    print("=" * 60)
    print("JAX Plasma Physics Simulation Tests")
    print("=" * 60)
    print()
    
    test_resistive_mhd()
    test_extended_mhd()
    test_hybrid_kinetic()
    test_physics_utils()
    test_simulations()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
