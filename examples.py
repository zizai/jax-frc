import jax.numpy as jnp
from resistive_mhd import run_simulation as run_resistive_mhd
from extended_mhd import run_simulation as run_extended_mhd
from hybrid_kinetic import run_simulation as run_hybrid_kinetic
from physics_utils import (
    compute_alfven_speed, compute_cyclotron_frequency, 
    compute_larmor_radius, compute_skin_depth, compute_beta
)

def example_resistive_mhd():
    """
    Example: Resistive MHD simulation for FRC formation.
    Demonstrates flux function evolution with Chodura resistivity.
    """
    print("=" * 60)
    print("Resistive MHD Simulation (Lamy Ridge Model)")
    print("=" * 60)
    
    final_psi, final_I_coil, history = run_resistive_mhd(
        steps=500,
        nr=64,
        nz=128,
        V_bank=1000.0,
        L_coil=1e-6,
        M_plasma_coil=1e-7
    )
    
    print(f"Final flux function max: {jnp.max(final_psi):.6f}")
    print(f"Final coil current: {final_I_coil:.6f} A")
    print(f"Simulation history shape: {history.shape}")
    print("\nPhysics: This model uses the Grad-Shafranov equation")
    print("with Chodura anomalous resistivity for rapid reconnection.")
    print("Best for: Circuit design and formation dynamics.")
    print("=" * 60)
    
    return final_psi, final_I_coil, history

def example_extended_mhd():
    """
    Example: Extended MHD simulation with Hall term.
    Demonstrates semi-implicit stepping for Whistler waves.
    """
    print("\n" + "=" * 60)
    print("Extended MHD Simulation (NIMROD Model)")
    print("=" * 60)
    
    b_final, history = run_extended_mhd(
        steps=100,
        nx=32,
        ny=32,
        dt=1e-6,
        eta=1e-4
    )
    
    print(f"Final B_x max: {jnp.max(b_final[0]):.6f}")
    print(f"Final B_y max: {jnp.max(b_final[1]):.6f}")
    print(f"Final B_z max: {jnp.max(b_final[2]):.6f}")
    print(f"Simulation history shape: {history[0].shape}")
    print("\nPhysics: This model includes the Hall term (J x B)/(ne)")
    print("and uses semi-implicit stepping to handle Whistler waves.")
    print("Best for: Global stability analysis and Hall effect studies.")
    print("=" * 60)
    
    return b_final, history

def example_hybrid_kinetic():
    """
    Example: Hybrid Kinetic-Fluid simulation with delta-f PIC.
    Demonstrates particle-based ion dynamics with fluid electrons.
    """
    print("\n" + "=" * 60)
    print("Hybrid Kinetic-Fluid Simulation (HYM Model)")
    print("=" * 60)
    
    x_final, v_final, w_final, history = run_hybrid_kinetic(
        steps=100,
        n_particles=1000,
        nr=32,
        nz=64,
        dt=1e-8,
        eta=1e-4
    )
    
    print(f"Number of particles: {x_final.shape[0]}")
    print(f"Particle positions shape: {x_final.shape}")
    print(f"Particle velocities shape: {v_final.shape}")
    print(f"Particle weights shape: {w_final.shape}")
    print(f"Weight statistics: mean={jnp.mean(w_final):.6f}, std={jnp.std(w_final):.6f}")
    print("\nPhysics: This model treats ions as particles (delta-f PIC)")
    print("and electrons as a fluid. Captures FLR effects and beam physics.")
    print("Best for: Stability limits and neutral beam injection studies.")
    print("=" * 60)
    
    return x_final, v_final, w_final, history

def example_physics_calculations():
    """
    Example: Physics calculations using utility functions.
    """
    print("\n" + "=" * 60)
    print("Physics Calculations")
    print("=" * 60)
    
    B = 1.0
    n = 1e19
    T = 100.0
    v_perp = 1e5
    
    v_A = compute_alfven_speed(B, n)
    omega_c = compute_cyclotron_frequency(B)
    r_L = compute_larmor_radius(v_perp, B)
    d_i = compute_skin_depth(n)
    beta = compute_beta(n, T, B)
    
    print(f"Alfvén speed: {v_A:.2e} m/s")
    print(f"Cyclotron frequency: {omega_c:.2e} rad/s")
    print(f"Larmor radius: {r_L:.2e} m")
    print(f"Ion skin depth: {d_i:.2e} m")
    print(f"Plasma beta: {beta:.4f}")
    print("\nThese parameters characterize the FRC plasma state.")
    print("=" * 60)
    
    return {
        'v_A': v_A,
        'omega_c': omega_c,
        'r_L': r_L,
        'd_i': d_i,
        'beta': beta
    }

def compare_models():
    """
    Compare the three models in terms of physics and computational cost.
    """
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    models = [
        {
            'name': 'Resistive MHD (Lamy Ridge)',
            'equations': 'E + v x B = eta J',
            'method': 'Flux Function / Finite Volume',
            'cost': 'Low (Minutes)',
            'stability': 'Fails (Predicts instability)',
            'use_case': 'Circuit & Coil Design'
        },
        {
            'name': 'Extended MHD (NIMROD)',
            'equations': 'E + v x B = eta J + (J x B)/(ne)',
            'method': 'Finite Element + Semi-Implicit',
            'cost': 'High (Hours/Days)',
            'stability': 'Good (Captures Hall effect)',
            'use_case': 'Global Dynamics & Thermal Transport'
        },
        {
            'name': 'Hybrid Kinetic (HYM)',
            'equations': 'E <- Fluid Closure; Ions <- Particles',
            'method': 'Delta-f Particle-in-Cell',
            'cost': 'Extreme (Days/Weeks)',
            'stability': 'Excellent (Captures FLR & Beams)',
            'use_case': 'Stability Limits & NBI Physics'
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"\nModel {i}: {model['name']}")
        print(f"  Equations: {model['equations']}")
        print(f"  Method: {model['method']}")
        print(f"  Cost: {model['cost']}")
        print(f"  FRC Stability: {model['stability']}")
        print(f"  Best Use Case: {model['use_case']}")
    
    print("\n" + "=" * 60)

def main():
    """
    Run all examples and comparisons.
    """
    print("\n" + "=" * 60)
    print("JAX Plasma Physics Simulation Examples")
    print("Based on Field-Reversed Configuration (FRC) Models")
    print("=" * 60)
    
    try:
        final_psi, final_I_coil, history = example_resistive_mhd()
    except Exception as e:
        print(f"Resistive MHD example failed: {e}")
    
    try:
        b_final, history = example_extended_mhd()
    except Exception as e:
        print(f"Extended MHD example failed: {e}")
    
    try:
        x_final, v_final, w_final, history = example_hybrid_kinetic()
    except Exception as e:
        print(f"Hybrid Kinetic example failed: {e}")
    
    try:
        physics_params = example_physics_calculations()
    except Exception as e:
        print(f"Physics calculations failed: {e}")
    
    compare_models()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
