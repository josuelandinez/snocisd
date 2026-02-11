import numpy as np
from pyscf import gto, scf, ci
from noci_jax import nocisd_general, slater_general

def run_system_test(name, atom_str, basis, spin_val):
    print("\n" + "="*80)
    print(f"TEST SYSTEM: {name}")
    print("="*80)
    
    # 1. Setup
    mol = gto.M(atom=atom_str, basis=basis, spin=spin_val, verbose=0)
    mf = scf.UHF(mol).run()
    
    print(f"UHF Energy:                 {mf.e_tot:.8f} Ha")
    print(f"Electrons: {mol.nelec} (Alpha={mol.nelec[0]}, Beta={mol.nelec[1]})")

    # 2. Reference UCISD
    myci = ci.UCISD(mf).run()
    e_cisd = myci.e_tot
    print(f"Reference UCISD Energy:     {e_cisd:.8f} Ha")
    
    # 3. NOCI Generation
    dt = 0.1
    print(f"Generating NOCI (dt={dt})...")
    ta_list, tb_list, c_analytic = nocisd_general.gen_nocisd_multiref_general(None, mf, dt=dt, tol=1e-8)
    rmats = (ta_list, tb_list)
    print(f"Determinants:               {len(ta_list)}")
    
    # 4. Energy Computation
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')
    e_nuc = mol.energy_nuc()
    
    # A. Analytic
    e_ana = slater_general.compute_energy_from_coeffs(rmats, mf.mo_coeff, h1e, h2e, e_nuc, c_analytic)
    
    # B. Variational
    e_var = slater_general.noci_energy(rmats, mf.mo_coeff, h1e, h2e, e_nuc, tol=1e-8)
    
    # 5. Results
    print("-" * 60)
    print(f"Reference UCISD:      {e_cisd:.8f} Ha")
    print(f"NOCI Analytic:        {e_ana:.8f} Ha")
    print(f"NOCI Variational:     {e_var:.8f} Ha")
    
    diff_ana = e_ana - e_cisd
    diff_var = e_var - e_cisd
    
    # Verification Logic
    print(f"\nAnalytic Deviation:   {diff_ana*1000:.3f} mHa")
    
    if abs(diff_ana) < 5.0: # Tolerance for finite difference error
        print("SUCCESS: Analytic construction matches UCISD topology.")
    else:
        print("WARNING: Analytic construction deviates significantly.")
        
    if e_var <= e_ana + 1e-8:
        print("SUCCESS: Variational optimization successful.")
        if e_var < e_cisd:
             print(f"BONUS: Captured {abs(diff_var*1000):.3f} mHa of higher-order correlation (Quadruples).")
    else:
        print("FAIL: Variational energy higher than Analytic.")

if __name__ == "__main__":
    # Test 1: Nitrogen (Rectangular AB Block Test)
    # N (High Spin): 5 Alpha, 2 Beta. AB block is 5x2 (or virtual equivalent).
    run_system_test("Nitrogen Atom (High Spin)", 'N 0 0 0', 'sto-3g', 3)
    
    # Test 2: Hydrogen Fluoride (Square Block Test)
    # HF (Closed Shell): 5 Alpha, 5 Beta. AA/BB/AB blocks are square.
    run_system_test("Hydrogen Fluoride (Closed Shell)", 'H 0 0 0; F 0 0 0.917', 'sto-3g', 0)
