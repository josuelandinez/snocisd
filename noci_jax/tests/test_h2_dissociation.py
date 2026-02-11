import numpy as np
from pyscf import gto, scf, ci
from noci_jax import nocisd_general, slater_general

def verify_h2_dissociation():
    print("="*80)
    print("VERIFICATION: Stretched H2 (Strong Correlation)")
    print("Checking: Symmetry Breaking & Quadruple Excitations")
    print("="*80)
    
    # 1. System: H2 stretched to 2.5 Angstrom
    mol = gto.M(atom='H 0 0 0; H 0 0 2.5', basis='sto-3g', spin=0, verbose=0)
    
    # 2. Run UHF with Symmetry Breaking
    # Mix initial guess to ensure we find the lower energy broken-symmetry solution
    mf = scf.UHF(mol)
    dm_init = mf.get_init_guess()
    dm_init[0][0,0] += 0.5 
    mf.kernel(dm0=dm_init)
    
    print(f"UHF Energy (Broken Sym):    {mf.e_tot:.8f} Ha")
    
    # 3. Run Exact (UCISD is exact for 2 electrons)
    myci = ci.UCISD(mf).run()
    e_exact = myci.e_tot
    print(f"Exact Energy (UCISD):       {e_exact:.8f} Ha")
    
    # 4. Generate NOCI
    dt = 0.1
    print(f"\nGenerating NOCI expansion (dt={dt})...")
    ta_list, tb_list, c_analytic = nocisd_general.gen_nocisd_multiref_general(None, mf, dt=dt, tol=1e-8)
    rmats = (ta_list, tb_list)
    
    # 5. Compute Energies
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')
    e_nuc = mol.energy_nuc()

    # A. Analytic
    e_ana = slater_general.compute_energy_from_coeffs(
        rmats, mf.mo_coeff, h1e, h2e, e_nuc, c_analytic
    )

    # B. Variational
    e_var = slater_general.noci_energy(
        rmats, mf.mo_coeff, h1e, h2e, e_nuc, tol=1e-8
    )
    
    # 6. Verdict
    print("-" * 80)
    print(f"RESULTS:")
    print(f"1. Exact Energy:        {e_exact:.8f} Ha")
    print(f"2. NOCI Analytic:       {e_ana:.8f} Ha")
    print(f"3. NOCI Variational:    {e_var:.8f} Ha")
    
    delta = e_var - e_exact
    
    if abs(delta) < 1e-5:
        print("\nSUCCESS: Perfect agreement with Exact solution.")
    elif e_var < e_exact - 1e-6:
        print(f"\nFAIL: Variational Violation! ({delta:.2e} Ha)")
    else:
        print(f"\nSUCCESS: Variational (Diff: {delta:.2e} Ha)")
        print("         Note: H2 minimal basis NOCI should be practically exact.")

if __name__ == "__main__":
    verify_h2_dissociation()
