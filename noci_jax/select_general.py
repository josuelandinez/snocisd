import numpy as np
import scipy.linalg
from noci_jax import slater_general

def filter_redundant_determinants(rmats, mo_coeff, h1e, h2e, e_nuc, metric_tol=1e-5):
    """
    Uses QR decomposition with column pivoting on the Overlap matrix 
    to robustly extract the linearly independent subset of determinants.
    """
    print(f"   -> Filtering combined expansion of {rmats[0].shape[0]} determinants...")
    smat, hmat = slater_general.build_noci_matrices(rmats, mo_coeff, h1e, h2e, e_nuc)
    s_np = np.array(smat)
    
    # QR with column pivoting.
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
