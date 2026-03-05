import jax
import jax.numpy as jnp
from res_adj import stable_adjugate

# Vectorize the adjugate function to apply over a batch of matrices
# in_axes=(0) means we map over the first dimension (the batch)
batch_stable_adjugate = jax.vmap(stable_adjugate, in_axes=(0,))

def build_all_couplings_opt(U_batch_A, U_batch_B):
    """
    U_batch_A: Array of shape (N_pairs, N_mo, N_occ)
    U_batch_B: Array of shape (N_pairs, N_mo, N_occ)
    Computes all pairwise adjugates in a single vectorized hardware operation.
    """
    # Batch matrix multiplication: (N_pairs, N_occ, N_occ)
    S_AB_batch = jnp.einsum('nij,njk->nik', jnp.transpose(U_batch_A, (0, 2, 1)), U_batch_B)
    
    # Compute adjugates simultaneously
    return batch_stable_adjugate(S_AB_batch)
