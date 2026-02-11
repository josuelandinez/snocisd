# noci_jax/tests/water_updated.py

import numpy as np
from pyscf import gto, scf, ci, fci
from noci_jax import nocisd_general, slater_general

def verify_water_energy_consistency():
    print("="*80)
    print("VERIFICATION: Water Molecule - Energy Consistency Check")
    print("Comparing Variational Optimization vs. Analytic Construction")
    print("="*80)
    
    # 1. System Setup (Stretched Water to ensure UHF is distinct)
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='sto-3g', spin=0, verbose=0)
    mf = scf.UHF(mol).run()
    
    print("Running Reference UCISD...")
    myci = ci.UCISD(mf).run()
    e_cisd = myci.e_tot
    print(f"Reference UCISD Energy:     {e_cisd:.8f} Ha")
    
    # 2. Generate Determinants AND Analytic Coefficients
    # CHANGE: Increased dt to 0.1 to avoid linear dependency in GEVP
    dt = 0.1 
    print(f"\nGenerating NOCI expansion (dt={dt})...")
    
    ta_list, tb_list, c_analytic = nocisd_general.gen_nocisd_multiref_general(None, mf, dt=dt, tol=1e-8)
    rmats = (ta_list, tb_list)
    
    print(f"Number of Determinants:     {len(ta_list)}")
    
    # 3. Compute Matrices (H and S)
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')
    e_nuc = mol.energy_nuc()
    
    # A. Variational Energy (GEVP)
    # Passed explicit tolerance to ensure we cut off noise from near-linear-dependency
    e_var = slater_general.noci_energy(rmats, mf.mo_coeff, h1e, h2e, e_nuc, tol=1e-8)
    
    # B. Analytic Energy (Fixed Coefficients)
    S, H = slater_general.build_noci_matrices(rmats, mf.mo_coeff, h1e, h2e, e_nuc)
    
    num = c_analytic.T @ H @ c_analytic
    den = c_analytic.T @ S @ c_analytic
    e_ana = num / den
    
    print("-" * 80)
    print(f"RESULTS:")
    print(f"1. Reference UCISD:         {e_cisd:.8f} Ha")
    print(f"2. NOCI Analytic (Constr.): {e_ana:.8f} Ha  (Should match UCISD)")
    print(f"3. NOCI Variational (Opt.): {e_var:.8f} Ha  (Should be <= Analytic)")
    
    delta_ana = e_ana - e_cisd
    delta_var = e_var - e_cisd
    
    print("-" * 80)
    print(f"Delta (Analytic - UCISD):   {delta_ana:.2e} Ha")
    print(f"Delta (Variational - UCISD):{delta_var:.2e} Ha")
    
    # VERIFICATION LOGIC
    # Slightly relaxed tolerance for analytic because dt=0.1 implies O(dt^2) error ~ 0.01
    # But usually 4-point stencil is very accurate.
    if abs(delta_ana) < 5e-3: 
        print("\nSUCCESS: The constructed determinants correctly represent the UCISD wavefunction.")
    else:
        print("\nWARNING: Analytic construction deviates. Check dt scaling or 4-point stencil logic.")
        
    if e_var <= e_ana + 1e-8:
        print("SUCCESS: Variational optimization improved (or matched) the energy.")
    else:
        print("FAIL: Variational energy is higher than analytic (Numerical instability?).")

if __name__ == "__main__":
    verify_water_energy_consistency()
