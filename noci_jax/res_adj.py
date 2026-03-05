import jax
import jax.numpy as jnp

@jax.jit
def stable_adjugate(M):
    """
    Computes the adjugate of matrix M safely.
    Instead of using SVD (which has undefined JVP gradients for degenerate 
    singular values like core orbitals), this uses the Cofactor Expansion (Minors).
    Since N_occ is small, this is highly efficient and 100% autodiff safe.
    """
    n = M.shape[0]
    
    # Build the cofactor matrix: C_ij = (-1)^(i+j) * det(Minor_ij)
    C_rows = []
    for i in range(n):
        C_cols = []
        for j in range(n):
            # Drop row i and column j to form the minor
            minor = jnp.delete(jnp.delete(M, i, axis=0), j, axis=1)
            sign = 1.0 if (i + j) % 2 == 0 else -1.0
            C_cols.append(sign * jnp.linalg.det(minor))
            
        C_rows.append(jnp.stack(C_cols))
        
    C = jnp.stack(C_rows)
    
    # The adjugate is the transpose of the cofactor matrix
    return C.T

def build_coupling_adj(U_A, U_B):
    S_AB = U_A.T @ U_B
    return stable_adjugate(S_AB)
