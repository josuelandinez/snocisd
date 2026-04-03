import numpy as np
import scipy.linalg
import jax.numpy as jnp
from noci_jax import slater_general

def filter_redundant_determinants(rmats, mo_coeff, h1e, h2e, e_nuc, metric_tol=1e-5):
    """
    Uses QR decomposition with column pivoting on the Overlap matrix 
    to robustly extract the linearly independent subset of determinants.
    """
    print(f"   -> Filtering combined expansion of {rmats[0].shape[0]} determinants...")
    smat, hmat = slater_general.build_noci_matrices(rmats, mo_coeff, h1e, h2e, e_nuc)
    s_np = np.array(smat)
    
    Q, R, P = scipy.linalg.qr(s_np, pivoting=True)
    R_diag = np.abs(np.diag(R))
    independent_count = np.sum(R_diag > metric_tol)
    keep_indices = np.sort(P[:independent_count])
    
    ra_filtered = rmats[0][keep_indices]
    rb_filtered = rmats[1][keep_indices]
    print(f"   -> Kept {independent_count} linearly independent determinants.")
    
    smat_filt = s_np[np.ix_(keep_indices, keep_indices)]
    hmat_filt = np.array(hmat)[np.ix_(keep_indices, keep_indices)]
    return (ra_filtered, rb_filtered), smat_filt, hmat_filt

def get_overlap_column(rmats, pivot_idx, block_size=500):
    """
    Computes a single column of the Global Overlap matrix matrix-free.
    """
    ra, rb = rmats
    N = ra.shape[0]
    ra_p = ra[pivot_idx:pivot_idx+1]
    rb_p = rb[pivot_idx:pivot_idx+1]
    
    S_col = np.zeros(N, dtype=np.float64)
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        ra_chunk, rb_chunk = ra[i:i_end], rb[i:i_end]
        
        met_a = jnp.einsum('nji, mjk -> nmik', ra_chunk.conj(), ra_p)
        det_a = jnp.linalg.det(met_a)[:, 0]
        met_b = jnp.einsum('nji, mjk -> nmik', rb_chunk.conj(), rb_p)
        det_b = jnp.linalg.det(met_b)[:, 0]
        S_col[i:i_end] = np.real(det_a * det_b)
        
    return S_col

def get_diagonal(rmats, block_size=500):
    """
    Computes the true self-overlap (diagonal of the Gram matrix) matrix-free.
    """
    ra, rb = rmats
    N = ra.shape[0]
    d = np.zeros(N, dtype=np.float64)
    
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        ra_chunk, rb_chunk = ra[i:i_end], rb[i:i_end]
        
        met_a = jnp.einsum('nji, njk -> nik', ra_chunk.conj(), ra_chunk)
        det_a = jnp.linalg.det(met_a)
        met_b = jnp.einsum('nji, njk -> nik', rb_chunk.conj(), rb_chunk)
        det_b = jnp.linalg.det(met_b)
        d[i:i_end] = np.real(det_a * det_b)
        
    return d

def matrix_free_pivoted_cholesky(rmats, metric_tol=1e-7, max_rank=2500, block_size=500):
    """
    Extracts the independent geometric basis via Pivoted Cholesky without 
    building the dense global Overlap matrix.
    """
    N = rmats[0].shape[0]
    d = get_diagonal(rmats, block_size)
    
    keep_indices = []
    L = np.zeros((N, max_rank), dtype=np.float64)
    
    print(f"   -> Starting Matrix-Free Cholesky on {N} determinants...")
    for k in range(max_rank):
        if k == 0:
            p = 0
        else:
            p = int(np.argmax(d))
            if d[p] < metric_tol:
                print(f"      Terminated early at rank {k}. Max residual: {d[p]:.2e}")
                break
                
        keep_indices.append(p)
        S_col = get_overlap_column(rmats, p, block_size)
        
        if k == 0:
            L[:, k] = S_col / np.sqrt(d[p])
        else:
            L[:, k] = (S_col - L[:, :k] @ L[p, :k]) / np.sqrt(d[p])
            
        d -= L[:, k]**2
        d = np.clip(d, 0.0, None)
        d[p] = 0.0
        
    print(f"   -> Cholesky complete. Safely extracted {len(keep_indices)} independent vectors.")
    return keep_indices

def filter_by_energy_sun_dutta_iterative(s_np, h_np, candidate_indices, thresholds=[1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12], metric_tol=1e-8):
    """
    Iterative Thresholding Sun-Dutta Filter. Cures Order Bias by capturing 
    strongest correlation first.
    """
    keep_indices = [0]
    pool = list(candidate_indices)
    if 0 in pool:
        pool.remove(0)
        
    smat_fix = s_np[[0], :][:, [0]]
    hmat_fix = h_np[[0], :][:, [0]]
    E_fix = hmat_fix[0, 0] / smat_fix[0, 0]
    noci_vec = np.array([1.0])
    
    print(f"   -> Beginning Iterative Sun-Dutta Sweeps...")
    
    for e_tol in thresholds:
        survivors_this_sweep = 0
        
        for i in list(pool):
            smat_left = s_np[keep_indices, i]
            s_new = s_np[i, i]
            hmat_left = h_np[keep_indices, i]
            h_new_raw = h_np[i, i]
            
            inv_fix = scipy.linalg.pinvh(smat_fix, atol=1e-12)
            proj_old = smat_left.conj().T @ inv_fix @ smat_left
            proj_new = 1.0 - proj_old / s_new
            
            if proj_new < metric_tol:
                pool.remove(i) 
                continue 
                
            norm_fix = noci_vec.conj().T @ smat_fix @ noci_vec
            H_fix = E_fix * norm_fix
            
            b_p = inv_fix @ smat_left
            T = noci_vec.conj().T @ (hmat_left - hmat_fix @ b_p)
            H_new = h_new_raw - 2 * np.real(b_p.conj().T @ hmat_left) + b_p.conj().T @ hmat_fix @ b_p
            norm_new = proj_new * s_new
            
            H_22 = np.array([[H_fix, T], [np.conj(T), H_new]])
            S_22 = np.array([[norm_fix, 0], [0, norm_new]])
            
            try:
                vals, vecs = scipy.linalg.eigh(H_22, S_22)
                epsilon = vals[0]
                vec = vecs[:, 0]
                ratio = (E_fix - epsilon) / abs(E_fix)
                
                if ratio > e_tol:
                    keep_indices.append(i)
                    pool.remove(i)
                    survivors_this_sweep += 1
                    
                    smat_fix = s_np[np.ix_(keep_indices, keep_indices)]
                    hmat_fix = h_np[np.ix_(keep_indices, keep_indices)]
                    E_fix = epsilon
                    
                    c = np.zeros(len(keep_indices))
                    c[:-1] = vec[0] * noci_vec - vec[1] * b_p
                    c[-1] = vec[1]
                    noci_vec = c
            except:
                continue
                
        print(f"      Threshold {e_tol:.0e}: Added {survivors_this_sweep} dets. Active Space: {len(keep_indices)}")
        if len(pool) == 0:
            break
            
    return keep_indices
