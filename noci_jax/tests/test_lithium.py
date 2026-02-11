import numpy as np
from pyscf import gto, scf, fci, ci
from noci_jax import nocisd_general, slater_general

def verify_lithium():
    print("="*80)
    print("VERIFICATION: Lithium Atom (Open-Shell Doublet)")
    print("Checking: UHF vs Exact FCI vs NOCI (Analytic & Variational)")
    print("="*80)

    # 1. Setup System
    # Using STO-3G for speed, but physics holds for cc-pVDZ too
    mol = gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
    mf = scf.UHF(mol).run()
    
    print(f"Hartree-Fock (UHF) Energy:    {mf.e_tot:.8f} Ha")
    print(f"Electrons: {mol.nelec} (Alpha={mol.nelec[0]}, Beta={mol.nelec[1]})")

    # 2. Run Exact Full CI (Gold Standard)
    myfci = fci.FCI(mf, mf.mo_coeff)
    e_fci, _ = myfci.kernel()
    print(f"Exact Full CI Energy:         {e_fci:.8f} Ha")

    # 3. Generate NOCI Expansion
    # dt=0.1 balances Taylor accuracy with numerical stability
    dt = 0.1
    print(f"\nGenerating NOCI expansion (dt={dt})...")
    
    # Passing None for t_ref triggers the Single-Reference default logic
    ta_list, tb_list, c_analytic = nocisd_general.gen_nocisd_multiref_general(None, mf, dt=dt, tol=1e-8)
    rmats = (ta_list, tb_list)
    print(f"Number of Determinants:       {len(ta_list)}")

    # 4. Compute Energies
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e')
    e_nuc = mol.energy_nuc()

    # A. Analytic Energy (Verification of Construction)
    e_ana = slater_general.compute_energy_from_coeffs(
        rmats, mf.mo_coeff, h1e, h2e, e_nuc, c_analytic
    )

    # B. Variational Energy (Optimization)
    e_var = slater_general.noci_energy(
        rmats, mf.mo_coeff, h1e, h2e, e_nuc, tol=1e-8
    )

    # 5. Results & Verdict
    print("-" * 80)
    print(f"RESULTS:")
    print(f"1. Exact Full CI:       {e_fci:.8f} Ha")
    print(f"2. NOCI Analytic:       {e_ana:.8f} Ha")
    print(f"3. NOCI Variational:    {e_var:.8f} Ha")

    delta_ana = e_ana - e_fci
    delta_var = e_var - e_fci

    print(f"\nErrors vs FCI:")
    print(f"Analytic Error:         {delta_ana*1000:.4f} mHa")
    print(f"Variational Error:      {delta_var*1000:.4f} mHa")

    # Success Criteria
    # NOCI should be strictly variational (>= FCI)
    # Analytic should be close to FCI (since Li has few electrons, CISD ~ FCI)
    if e_var < e_fci - 1e-6:
        print("\nFAIL: Variational Violation (Energy lower than Exact)")
    elif e_var > mf.e_tot:
        print("\nFAIL: No correlation captured (Higher than UHF)")
    else:
        print("\nSUCCESS: NOCI is Variational and Correlated.")
        if e_var < e_ana:
            print("         Variational optimization improved upon the analytic construction.")

if __name__ == "__main__":
    verify_lithium()
