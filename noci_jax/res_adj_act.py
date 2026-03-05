import jax.numpy as jnp
from res_adj import stable_adjugate

def build_coupling_act(U_A, U_B, n_core):
    """
    Builds the adjugate coupling matrix exploiting the frozen core.
    U_A, U_B are shape (N_mo, N_occ).
    """
    # Split into Core and Active parts
    C_A = U_A[:, :n_core]
    Act_A = U_A[:, n_core:]
    Act_B = U_B[:, n_core:]
    
    # Overlap of the active space only
    S_act = Act_A.T @ Act_B
    
    # Compute adjugate and determinant of the much smaller active block
    adj_S_act = stable_adjugate(S_act)
    det_S_act = jnp.linalg.det(S_act)
    
    n_act = S_act.shape[0]
    
    # Reconstruct the full effective adjugate
    # Core block gets scaled by the determinant of the active block
    adj_full = jnp.block([
        [jnp.eye(n_core) * det_S_act, jnp.zeros((n_core, n_act))],
        [jnp.zeros((n_act, n_core)),  adj_S_act]
    ])
    
    return adj_full
