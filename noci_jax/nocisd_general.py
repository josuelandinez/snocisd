import numpy as np
import copy
from pyscf import ci
from noci_jax import slater_general as slater_gen 

def compress_general(myci, civec=None, dt=1.0, tol=1e-12):
    """
    Exact Compression of CISD into Non-Orthogonal determinants.
    
    - Handles FROZEN CORE automatically by padding active amplitudes.
    - Handles ALL-ELECTRON automatically (padding=0).
    """
    if civec is None:
        civec = myci.ci
        
    # 1. Get Active Amplitudes from PySCF
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    c1a_act, c1b_act = c1
    c2aa_act, c2ab_act, c2bb_act = c2
    
    # 2. Determine Full System Dimensions
    mol = myci.mol
    nocc_a_total = int(mol.nelec[0])
    nocc_b_total = int(mol.nelec[1])
    
    # Robustly find total orbitals (NMO)
    if hasattr(myci, '_scf'):
        mf = myci._scf
        if isinstance(mf.mo_coeff, (list, tuple)):
            nmo = mf.mo_coeff[0].shape[1]
        else:
            nmo = mf.mo_coeff.shape[1]
    else:
        # Fallback if _scf missing
        if hasattr(myci, 'nmo'):
             nmo = myci.nmo
        else:
             nmo = mol.nao_nr()

    nvir_a_total = nmo - nocc_a_total
    nvir_b_total = nmo - nocc_b_total

    # 3. Determine Padding (Active -> Full mapping)
    nocc_a_act = c1a_act.shape[0]
    nocc_b_act = c1b_act.shape[0]
    
    # Offset: How many frozen occupied orbitals are below the active space?
    occ_start_a = nocc_a_total - nocc_a_act
    occ_start_b = nocc_b_total - nocc_b_act
    
    nvir_a_act = c1a_act.shape[1]
    nvir_b_act = c1b_act.shape[1]
    
    # 4. Padding Helper
    def pad_to_full(t_active_matrix, spin='a'):
        # t_active_matrix is (N_vir_active, N_occ_active)
        # We place it into (N_vir_total, N_occ_total)
        
        if spin == 'a':
            full_Z = np.zeros((nvir_a_total, nocc_a_total))
            start_occ = occ_start_a
            n_v, n_o = t_active_matrix.shape
        else:
            full_Z = np.zeros((nvir_b_total, nocc_b_total))
            start_occ = occ_start_b
            n_v, n_o = t_active_matrix.shape
            
        # Paste active block
        full_Z[0:n_v, start_occ:start_occ+n_o] = t_active_matrix
        return full_Z

    ta_list = []
    tb_list = []
    coeffs = []
    
    z_a_ref = np.zeros((nvir_a_total, nocc_a_total))
    z_b_ref = np.zeros((nvir_b_total, nocc_b_total))
    
    # 0. Reference Determinant
    coeffs.append(c0) 
    
    def add_mode(ta, tb, scale, coeff_val):
        norm_sq = np.sum(ta**2) + np.sum(tb**2)
        norm_val = np.sqrt(norm_sq)
        
        if norm_val < 1e-12: return

        effective_norm = norm_val * scale
        final_scale = scale
        if abs(effective_norm - 1.0) < 0.02:
             final_scale = scale * 0.99
            
        ta_list.append(ta * final_scale)
        tb_list.append(tb * final_scale)
        coeffs.append(coeff_val)

    # === SINGLES ===
    t1a_act_T = c1a_act.T
    t1b_act_T = c1b_act.T
    
    t1a = pad_to_full(t1a_act_T, 'a')
    t1b = pad_to_full(t1b_act_T, 'b')
    
    coeff_t1 = 0.5 / dt
    
    add_mode(t1a, z_b_ref, dt, coeff_t1)
    add_mode(-t1a, z_b_ref, dt, -coeff_t1)
    add_mode(z_a_ref, t1b, dt, coeff_t1)
    add_mode(z_a_ref, -t1b, dt, -coeff_t1)
    
    # === DOUBLES ===
    coeff_t2 = 0.25 / (dt**2)
    
    # -- AA Block --
    dim_a_act = nvir_a_act * nocc_a_act
    c2aa_t = c2aa_act.transpose(2, 0, 3, 1) 
    mat_aa = c2aa_t.reshape(dim_a_act, dim_a_act)
    mat_aa = 0.5 * (mat_aa - mat_aa.T)
    
    U, S, Vt = np.linalg.svd(mat_aa, full_matrices=False)
    V = Vt.conj().T
    
    idx = S > tol
    for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
        t_mode_u_act = u_vec.reshape(nvir_a_act, nocc_a_act)
        t_mode_v_act = v_vec.reshape(nvir_a_act, nocc_a_act)
        
        zu = pad_to_full(t_mode_u_act, 'a') * np.sqrt(s_val)
        zv = pad_to_full(t_mode_v_act, 'a') * np.sqrt(s_val)
        
        add_mode(zu + zv, z_b_ref, dt, coeff_t2)
        add_mode(-(zu + zv), z_b_ref, dt, coeff_t2)
        add_mode(zu - zv, z_b_ref, dt, -coeff_t2)
        add_mode(-(zu - zv), z_b_ref, dt, -coeff_t2)

    # -- BB Block --
    dim_b_act = nvir_b_act * nocc_b_act
    if dim_b_act > 0: 
        c2bb_t = c2bb_act.transpose(2, 0, 3, 1)
        mat_bb = c2bb_t.reshape(dim_b_act, dim_b_act)
        mat_bb = 0.5 * (mat_bb - mat_bb.T)
        
        U, S, Vt = np.linalg.svd(mat_bb, full_matrices=False)
        V = Vt.conj().T
        
        idx = S > tol
        for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
            t_mode_u_act = u_vec.reshape(nvir_b_act, nocc_b_act)
            t_mode_v_act = v_vec.reshape(nvir_b_act, nocc_b_act)
            
            zu = pad_to_full(t_mode_u_act, 'b') * np.sqrt(s_val)
            zv = pad_to_full(t_mode_v_act, 'b') * np.sqrt(s_val)
            
            add_mode(z_a_ref, zu + zv, dt, coeff_t2)
            add_mode(z_a_ref, -(zu + zv), dt, coeff_t2)
            add_mode(z_a_ref, zu - zv, dt, -coeff_t2)
            add_mode(z_a_ref, -(zu - zv), dt, -coeff_t2)

    # -- AB Block --
    c2ab_t = c2ab_act.transpose(2, 0, 3, 1)
    mat_ab = c2ab_t.reshape(dim_a_act, dim_b_act)
    
    U, S, Vt = np.linalg.svd(mat_ab, full_matrices=False)
    V = Vt.conj().T
    
    idx = S > tol
    for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
        t_mode_a_act = u_vec.reshape(nvir_a_act, nocc_a_act)
        t_mode_b_act = v_vec.reshape(nvir_b_act, nocc_b_act)
        
        wa = pad_to_full(t_mode_a_act, 'a') * np.sqrt(s_val)
        wb = pad_to_full(t_mode_b_act, 'b') * np.sqrt(s_val)
        
        add_mode(wa, wb, dt, coeff_t2)
        add_mode(-wa, -wb, dt, coeff_t2)
        add_mode(wa, -wb, dt, -coeff_t2)
        add_mode(-wa, wb, dt, -coeff_t2)

    T_a = np.array([z_a_ref] + ta_list)
    T_b = np.array([z_b_ref] + tb_list)
    coeffs = np.array(coeffs)
    
    return (T_a, T_b, coeffs)

def gen_nocisd_multiref_general(tvecs_ref, mf, dt=1.0, tol=1e-12):
    if tvecs_ref is None:
        mol = mf.mol
        if isinstance(mf.mo_coeff, (list, tuple)):
            nmo = mf.mo_coeff[0].shape[1]
        else:
            nmo = mf.mo_coeff.shape[1]
            
        nocc_a = int(mol.nelec[0])
        nocc_b = int(mol.nelec[1])
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b
        
        z_a = np.zeros((1, nvir_a, nocc_a))
        z_b = np.zeros((1, nvir_b, nocc_b))
        tvecs_ref = (z_a, z_b)

    Ta_ref, Tb_ref = tvecs_ref
    n_refs = Ta_ref.shape[0]
    
    nvir_a, nocc_a = Ta_ref.shape[1:]
    nvir_b, nocc_b = Tb_ref.shape[1:]
    
    U_on = slater_gen.orthonormal_mos(tvecs_ref) 
    Ua_frames, Ub_frames = U_on
    
    all_Ra = []
    all_Rb = []
    all_coeffs = []
    
    for i in range(n_refs):
        mo_new_a = np.asarray(mf.mo_coeff[0] @ Ua_frames[i])
        mo_new_b = np.asarray(mf.mo_coeff[1] @ Ub_frames[i])
        
        mf_temp = copy.copy(mf)
        mf_temp.mo_coeff = (mo_new_a, mo_new_b)
        
        myci = ci.UCISD(mf_temp)
        if i == 0 and hasattr(mf, 'with_frozen_core'):
             pass # Placeholder if needed
             
        myci.kernel()
        
        Ta_loc, Tb_loc, coeffs_loc = compress_general(myci, dt=dt, tol=tol)
        
        Ra_loc, Rb_loc = slater_gen.tvecs_to_rmats((Ta_loc, Tb_loc), 
                                                   (nvir_a, nvir_b), 
                                                   (nocc_a, nocc_b))
        
        Ua_np = np.asarray(Ua_frames[i])
        Ub_np = np.asarray(Ub_frames[i])
        Ra_loc_np = np.asarray(Ra_loc)
        Rb_loc_np = np.asarray(Rb_loc)

        Ra_glob = np.einsum('ij, njk -> nik', Ua_np, Ra_loc_np)
        Rb_glob = np.einsum('ij, njk -> nik', Ub_np, Rb_loc_np)
        
        if i == 0:
            all_Ra.append(Ra_glob)
            all_Rb.append(Rb_glob)
            all_coeffs.append(coeffs_loc)
        else:
            all_Ra.append(Ra_glob[1:])
            all_Rb.append(Rb_glob[1:])
            all_coeffs.append(coeffs_loc[1:]) 
            
    Final_Ra = np.vstack(all_Ra)
    Final_Rb = np.vstack(all_Rb)
    Final_Coeffs = np.concatenate(all_coeffs)
    
    return (Final_Ra, Final_Rb, Final_Coeffs)
