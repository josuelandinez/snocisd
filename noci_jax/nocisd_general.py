import numpy as np
import copy
from pyscf import ci
from noci_jax import slater_general as slater_gen 

def compress_general(myci, civec=None, dt=1.0, tol=1e-12):
    """
    Exact Compression of CISD into Non-Orthogonal determinants.
    
    Returns:
    - T_a, T_b: Lists of rotation matrices.
    - coeffs: Analytic coefficients for reconstruction.
    """
    if civec is None:
        civec = myci.ci
        
    c0, c1, c2 = myci.cisdvec_to_amplitudes(civec)
    c1a, c1b = c1
    c2aa, c2ab, c2bb = c2
    
    nocc_a, nvir_a = c1a.shape
    nocc_b, nvir_b = c1b.shape
    
    ta_list = []
    tb_list = []
    coeffs = []
    
    z_a = np.zeros((nvir_a, nocc_a))
    z_b = np.zeros((nvir_b, nocc_b))
    
    # 0. Reference Determinant
    # We track the reference coefficient here to align with the determinant list structure
    coeffs.append(c0) 
    
    # Helper to add determinants with safety checks
    def add_mode(ta, tb, scale, coeff_val):
        norm_sq = np.sum(ta**2) + np.sum(tb**2)
        norm_val = np.sqrt(norm_sq)
        
        # 1. Filter numerical noise
        if norm_val < 1e-12:
            return

        # 2. Safety for H2 Singularity (when norm*dt ~ 1.0)
        effective_norm = norm_val * scale
        final_scale = scale
        
        if abs(effective_norm - 1.0) < 0.02:
             final_scale = scale * 0.99
            
        ta_list.append(ta * final_scale)
        tb_list.append(tb * final_scale)
        coeffs.append(coeff_val)

    # 1. Singles (T1)
    # Stencil: 1/(2*dt) * (|Phi+> - |Phi->)
    t1a = c1a.T 
    t1b = c1b.T 
    
    coeff_t1 = 0.5 / dt
    
    # Alpha Singles
    add_mode(t1a, z_b, dt, coeff_t1)
    add_mode(-t1a, z_b, dt, -coeff_t1)
    
    # Beta Singles
    add_mode(z_a, t1b, dt, coeff_t1)
    add_mode(z_a, -t1b, dt, -coeff_t1)
    
    # 2. Doubles Same Spin (AA) - ANTISYMMETRIC
    # Stencil: 1/(4*dt^2)
    coeff_t2 = 0.25 / (dt**2)
    
    dim_a = nvir_a * nocc_a
    c2aa_t = c2aa.transpose(2, 0, 3, 1) 
    mat_aa = c2aa_t.reshape(dim_a, dim_a)
    
    mat_aa = 0.5 * (mat_aa - mat_aa.T)
    
    U, S, Vt = np.linalg.svd(mat_aa, full_matrices=False)
    V = Vt.conj().T
    
    idx = S > tol
    for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
        t_mode_u = u_vec.reshape(nvir_a, nocc_a)
        t_mode_v = v_vec.reshape(nvir_a, nocc_a)
        
        factor = np.sqrt(s_val)
        
        zu = t_mode_u * factor
        zv = t_mode_v * factor
        
        # Q+ = zu + zv
        add_mode(zu + zv, z_b, dt, coeff_t2)
        add_mode(-(zu + zv), z_b, dt, coeff_t2)
        
        # Q- = zu - zv (minus sign in coeff for difference axis)
        add_mode(zu - zv, z_b, dt, -coeff_t2)
        add_mode(-(zu - zv), z_b, dt, -coeff_t2)

    # 3. Doubles Same Spin (BB) - ANTISYMMETRIC
    dim_b = nvir_b * nocc_b
    if dim_b > 0: 
        c2bb_t = c2bb.transpose(2, 0, 3, 1)
        mat_bb = c2bb_t.reshape(dim_b, dim_b)
        mat_bb = 0.5 * (mat_bb - mat_bb.T)
        
        U, S, Vt = np.linalg.svd(mat_bb, full_matrices=False)
        V = Vt.conj().T
        
        idx = S > tol
        for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
            t_mode_u = u_vec.reshape(nvir_b, nocc_b)
            t_mode_v = v_vec.reshape(nvir_b, nocc_b)
            
            factor = np.sqrt(s_val)
            
            zu = t_mode_u * factor
            zv = t_mode_v * factor
            
            add_mode(z_a, zu + zv, dt, coeff_t2)
            add_mode(z_a, -(zu + zv), dt, coeff_t2)
            add_mode(z_a, zu - zv, dt, -coeff_t2)
            add_mode(z_a, -(zu - zv), dt, -coeff_t2)

    # 4. Doubles Mixed Spin (AB) - GENERAL
    c2ab_t = c2ab.transpose(2, 0, 3, 1)
    mat_ab = c2ab_t.reshape(dim_a, dim_b)
    
    U, S, Vt = np.linalg.svd(mat_ab, full_matrices=False)
    V = Vt.conj().T
    
    idx = S > tol
    for s_val, u_vec, v_vec in zip(S[idx], U.T[idx], V.T[idx]):
        t_mode_a = u_vec.reshape(nvir_a, nocc_a)
        t_mode_b = v_vec.reshape(nvir_b, nocc_b)
        
        factor = np.sqrt(s_val)
        wa = t_mode_a * factor
        wb = t_mode_b * factor
        
        # Balanced Quartet for AB:
        add_mode(wa, wb, dt, coeff_t2)
        add_mode(-wa, -wb, dt, coeff_t2)
        add_mode(wa, -wb, dt, -coeff_t2)
        add_mode(-wa, wb, dt, -coeff_t2)

    T_a = np.array([z_a] + ta_list)
    T_b = np.array([z_b] + tb_list)
    coeffs = np.array(coeffs)
    
    return (T_a, T_b, coeffs)

def gen_nocisd_multiref_general(tvecs_ref, mf, dt=1.0, tol=1e-12):
    """
    Multi-reference extension.
    Handles None input for tvecs_ref by creating a standard HF reference.
    """
    # === NEW: Handle Single Reference Case ===
    if tvecs_ref is None:
        mol = mf.mol
        nmo = mf.mo_coeff[0].shape[1]
        nocc_a = int(mol.nelec[0])
        nocc_b = int(mol.nelec[1])
        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b
        
        # Create zero amplitudes (HF Reference)
        z_a = np.zeros((1, nvir_a, nocc_a))
        z_b = np.zeros((1, nvir_b, nocc_b))
        tvecs_ref = (z_a, z_b)
    # =========================================

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
        myci.kernel()
        
        # Unpack 3 values
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
