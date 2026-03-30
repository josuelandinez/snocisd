import numpy as np
import copy
from pyscf import ci
from noci_jax import slater_general as slater_gen 

def compress_general(myci, civec=None, dt=1.0, tol=1e-12):
    """
    Exact Compression of CISD into Non-Orthogonal determinants.
    """
    if civec is None:
        civec = myci.ci
        
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    c1a_act, c1b_act = c1
    c2aa_act, c2ab_act, c2bb_act = c2
    
    mol = myci.mol
    nocc_a_total = int(mol.nelec[0])
    nocc_b_total = int(mol.nelec[1])
    nmo = myci._scf.mo_coeff[0].shape[1] if hasattr(myci, '_scf') else mol.nao_nr()

    nvir_a_total, nvir_b_total = nmo - nocc_a_total, nmo - nocc_b_total
    nocc_a_act, nocc_b_act = c1a_act.shape[0], c1b_act.shape[0]
    occ_start_a, occ_start_b = nocc_a_total - nocc_a_act, nocc_b_total - nocc_b_act
    nvir_a_act, nvir_b_act = c1a_act.shape[1], c1b_act.shape[1]
    
    def pad_to_full(t_active_matrix, spin='a'):
        full_Z = np.zeros((nvir_a_total, nocc_a_total) if spin == 'a' else (nvir_b_total, nocc_b_total))
        start_occ = occ_start_a if spin == 'a' else occ_start_b
        n_v, n_o = t_active_matrix.shape
        full_Z[0:n_v, start_occ:start_occ+n_o] = t_active_matrix
        return full_Z

    # We need a mutable c0 reference since we will aggressively subtract from it
    c0_updated = c0

    ta_list, tb_list, coeffs = [], [], []
    z_a_ref = np.zeros((nvir_a_total, nocc_a_total))
    z_b_ref = np.zeros((nvir_b_total, nocc_b_total))
    
    def add_mode(ta, tb, scale, coeff_val):
        norm_val = np.sqrt(np.sum(ta**2) + np.sum(tb**2))
        if norm_val < 1e-12: return
        final_scale = scale * 0.99 if abs(norm_val * scale - 1.0) < 0.02 else scale
        ta_list.append(ta * final_scale)
        tb_list.append(tb * final_scale)
        coeffs.append(coeff_val)

    # === SINGLES ===
    coeff_t1 = 0.5 / dt
    add_mode(pad_to_full(c1a_act.T, 'a'), z_b_ref, dt, coeff_t1)
    add_mode(-pad_to_full(c1a_act.T, 'a'), z_b_ref, dt, -coeff_t1)
    add_mode(z_a_ref, pad_to_full(c1b_act.T, 'b'), dt, coeff_t1)
    add_mode(z_a_ref, -pad_to_full(c1b_act.T, 'b'), dt, -coeff_t1)
    
    # === DOUBLES ===
    c2aa_act = c2aa_act / 4.0
    c2bb_act = c2bb_act / 4.0
    
    # -- AA Block (Symmetric Matrix -> 2-Point Stencil) --
    coeff_t2_2pt = 1.0 / (dt**2)
    dim_a_act = nvir_a_act * nocc_a_act
    mat_aa = c2aa_act.transpose(2, 0, 3, 1).reshape(dim_a_act, dim_a_act)
    
    vals, vecs = np.linalg.eigh(mat_aa)
    idx = np.abs(vals) > tol
    for s_val, v_vec in zip(vals[idx], vecs[:, idx].T):
        zv = pad_to_full(v_vec.reshape(nvir_a_act, nocc_a_act), 'a') * np.sqrt(np.abs(s_val))
        c_mode = coeff_t2_2pt * np.sign(s_val)
        
        add_mode(zv, z_b_ref, dt, c_mode)
        add_mode(-zv, z_b_ref, dt, c_mode)
        
        # CRITICAL FIX: The 2-point stencil requires subtracting 2 * c_mode from the HF state!
        c0_updated -= 2.0 * c_mode

    # -- BB Block (Symmetric Matrix -> 2-Point Stencil) --
    dim_b_act = nvir_b_act * nocc_b_act
    if dim_b_act > 0: 
        mat_bb = c2bb_act.transpose(2, 0, 3, 1).reshape(dim_b_act, dim_b_act)
        vals, vecs = np.linalg.eigh(mat_bb)
        idx = np.abs(vals) > tol
        for s_val, v_vec in zip(vals[idx], vecs[:, idx].T):
            zv = pad_to_full(v_vec.reshape(nvir_b_act, nocc_b_act), 'b') * np.sqrt(np.abs(s_val))
            c_mode = coeff_t2_2pt * np.sign(s_val)
            
            add_mode(z_a_ref, zv, dt, c_mode)
            add_mode(z_a_ref, -zv, dt, c_mode)
            
            # CRITICAL FIX
            c0_updated -= 2.0 * c_mode

    # -- AB Block (Mixed Spin SVD -> 4-Point Stencil) --
    coeff_t2_4pt = 0.25 / (dt**2)
    mat_ab = c2ab_act.transpose(2, 0, 3, 1).reshape(dim_a_act, dim_b_act)
    
    U, S, Vt = np.linalg.svd(mat_ab, full_matrices=False)
    idx = S > tol
    for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], Vt.conj()):
        wa = pad_to_full(u_vec.reshape(nvir_a_act, nocc_a_act), 'a') * np.sqrt(s_val)
        wb = pad_to_full(v_vec.reshape(nvir_b_act, nocc_b_act), 'b') * np.sqrt(s_val)
        
        c_mode = coeff_t2_4pt
        add_mode(wa, wb, dt, c_mode)
        add_mode(-wa, -wb, dt, c_mode)
        add_mode(wa, -wb, dt, -c_mode)
        add_mode(-wa, wb, dt, -c_mode)
        
        # NO HF SHIFT REQUIRED for 4-point stencil, the difference operators cancel it naturally.

    # Prepend the Reference State with the vastly modified c0_updated
    T_a = np.array([z_a_ref] + ta_list)
    T_b = np.array([z_b_ref] + tb_list)
    all_coeffs = np.array([c0_updated] + coeffs)
    
    return (T_a, T_b, all_coeffs)

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
