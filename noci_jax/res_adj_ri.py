import jax
import jax.numpy as jnp
import numpy as np
from pyscf import df
from noci_jax.res_adj import stable_adjugate

def get_l_tensor_from_pyscf(mol, auxbasis='weigend'):
    """Extracts the 3-index Resolution of the Identity (RI) tensor from PySCF."""
    print(f"   -> Building PySCF Density Fitting (DF) object with basis: {auxbasis}")
    mydf = df.DF(mol)
    mydf.auxbasis = auxbasis
    mydf.build()
    
    cderi = mydf._cderi 
    nao = mol.nao_nr()
    naux = cderi.shape[0]
    
    L_tensor_np = np.zeros((naux, nao, nao))
    tril_idx = np.tril_indices(nao)
    
    for p in range(naux):
        L_tensor_np[p][tril_idx] = cderi[p]
        L_tensor_np[p] = L_tensor_np[p] + L_tensor_np[p].T - np.diag(np.diag(L_tensor_np[p]))
        
    return jnp.array(L_tensor_np)

def build_ri_jk_adj(L_tensor, P_adj):
    """Builds Coulomb (J) and Exchange (K) matrices using RI and adjugate density."""
    # Coulomb Matrix (J)
    J_aux = jnp.einsum('Prs,sr->P', L_tensor, P_adj)
    J_adj = jnp.einsum('P,Pij->ij', J_aux, L_tensor)
    
    # Exchange Matrix (K)
    tmp_K = jnp.einsum('Pir,rs->Pis', L_tensor, P_adj)
    K_adj = jnp.einsum('Pis,Pjs->ij', tmp_K, L_tensor)
    
    return J_adj, K_adj

def compute_noci_matrix_element_ri(U_A, U_B, S_ao, h1e, L_tensor):
    """Computes stable NOCI matrix element <A|H|B> using RI and adjugate."""
    S_AB = U_A.T @ S_ao @ U_B
    adj_S = stable_adjugate(S_AB)
    
    P_adj = U_B @ adj_S @ U_A.T
    E_1e_unnorm = jnp.sum(h1e * P_adj.T)
    
    J_adj, K_adj = build_ri_jk_adj(L_tensor, P_adj.T)
    G_adj = J_adj - 0.5 * K_adj
    E_2e_unnorm = 0.5 * jnp.sum(G_adj * P_adj.T)
    
    return E_1e_unnorm + E_2e_unnorm
