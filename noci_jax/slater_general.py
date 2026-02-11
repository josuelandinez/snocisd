# noci_jax/slater_general.py

import jax.numpy as jnp
import jax
from jax import config

# ENABLE x64 PRECISION
config.update("jax_enable_x64", True)

def tvecs_to_rmats(tvecs, nvir, nocc):
    ta, tb = tvecs
    na_vir, nb_vir = nvir
    na_occ, nb_occ = nocc
    N = ta.shape[0]

    # Alpha
    Ia = jnp.eye(na_occ)
    Ia = jnp.tile(Ia, (N, 1, 1))
    ra = jnp.concatenate([Ia, ta], axis=1)

    # Beta
    Ib = jnp.eye(nb_occ)
    Ib = jnp.tile(Ib, (N, 1, 1))
    rb = jnp.concatenate([Ib, tb], axis=1)

    return (ra, rb)

def build_noci_matrices(rmats, mo_coeff, h1e, h2e, e_nuc=0.0):
    '''
    Constructs the Hamiltonian (H) and Overlap (S) matrices 
    for the Non-Orthogonal Configuration Interaction.
    '''
    ra, rb = rmats
    Ca, Cb = mo_coeff[0], mo_coeff[1]

    # 1. Overlap Matrix (S)
    met_a = jnp.einsum('nji, mjk -> nmik', ra.conj(), ra)
    det_a = jnp.linalg.det(met_a)
    met_b = jnp.einsum('nji, mjk -> nmik', rb.conj(), rb)
    det_b = jnp.linalg.det(met_b)
    
    smat = det_a * det_b 
    smat = 0.5 * (smat + smat.T)

    # 2. Transition Density Matrices (P)
    inv_a = jnp.linalg.inv(met_a)
    inv_b = jnp.linalg.inv(met_b)

    sdets_a = jnp.einsum("ij, njk -> nik", Ca, ra)
    sdets_b = jnp.einsum("ij, njk -> nik", Cb, rb)

    trdm_a = jnp.einsum("mij, nmjk, nlk -> nmil", sdets_a, inv_a, sdets_a.conj())
    trdm_b = jnp.einsum("mij, nmjk, nlk -> nmil", sdets_b, inv_b, sdets_b.conj())
    
    trdm_tot = trdm_a + trdm_b

    # 3. Energy Contraction
    E1 = jnp.einsum("ij, nmji -> nm", h1e, trdm_tot)

    J_aa = jnp.einsum("ijkl, nmlk, nmji -> nm", h2e, trdm_a, trdm_a)
    J_bb = jnp.einsum("ijkl, nmlk, nmji -> nm", h2e, trdm_b, trdm_b)
    J_ab = jnp.einsum("ijkl, nmlk, nmji -> nm", h2e, trdm_b, trdm_a) 
    
    K_aa = jnp.einsum("ijkl, nmjk, nmli -> nm", h2e, trdm_a, trdm_a)
    K_bb = jnp.einsum("ijkl, nmjk, nmli -> nm", h2e, trdm_b, trdm_b)
    
    E2 = 0.5 * (J_aa - K_aa + J_bb - K_bb + 2 * J_ab)

    hmat = (E1 + E2) * smat
    hmat = 0.5 * (hmat + hmat.T)
    
    # Add nuclear repulsion to H
    hmat = hmat + smat * e_nuc

    return smat, hmat

def noci_energy(rmats, mo_coeff, h1e, h2e, e_nuc=0.0, tol=1e-10):
    '''
    General NOCI Energy.
    Standard Real-Valued Variational Solver.
    '''
    smat, hmat = build_noci_matrices(rmats, mo_coeff, h1e, h2e, e_nuc)
    return solve_lc_coeffs(hmat, smat, tol=tol)

def compute_energy_from_coeffs(rmats, mo_coeff, h1e, h2e, e_nuc, coeffs, tol=1e-10):
    '''
    Computes the expectation value E = <Psi|H|Psi> / <Psi|Psi>
    using a fixed set of coefficients.
    '''
    smat, hmat = build_noci_matrices(rmats, mo_coeff, h1e, h2e, e_nuc)
    
    c_vec = jnp.array(coeffs)
    
    # E = c.T @ H @ c / c.T @ S @ c
    num = c_vec.T @ hmat @ c_vec
    den = c_vec.T @ smat @ c_vec
    
    return num / den

def solve_lc_coeffs(hmat, smat, tol=1e-8):
    '''
    Robust Generalized Eigenvalue Solver.
    
    Crucial fix: Increased default tolerance to 1e-8 to handle
    linear dependencies introduced by small dt stencils.
    '''
    s_eig, s_vec = jnp.linalg.eigh(smat)
    
    # Filter small eigenvalues (Linear Dependency)
    keep_mask = s_eig > tol
    s_eig_kept = s_eig[keep_mask]
    s_vec_kept = s_vec[:, keep_mask]
    
    if s_eig_kept.shape[0] == 0:
         # Fallback if all eigenvalues are small (should not happen with Ref det)
         return jnp.mean(jnp.diag(hmat))

    # Canonical Orthogonalization
    inv_sqrt_s = 1.0 / jnp.sqrt(s_eig_kept)
    X = s_vec_kept @ jnp.diag(inv_sqrt_s)
    
    h_prime = X.conj().T @ hmat @ X
    vals, _ = jnp.linalg.eigh(h_prime)
    
    return vals[0]

# ... [Keep rotate_rmats and orthonormal_mos as they were] ...
def rotate_rmats(rmats, U):
    ra, rb = rmats
    Ua, Ub = U
    ra_new = jnp.einsum('ij, njk -> nik', Ua, ra)
    rb_new = jnp.einsum('ij, njk -> nik', Ub, rb)
    return (ra_new, rb_new)

def orthonormal_mos_spin(tmats):
    tshape = tmats.shape
    nvir, nocc = tshape[-2:]
    Nt = tmats.shape[0]
    
    Iocc = jnp.eye(nocc)
    Ivir = jnp.eye(nvir)

    Mocc = jnp.tile(Iocc, (Nt, 1, 1)) + jnp.einsum('nji, njk -> nik', tmats.conj(), tmats)
    Mvir = jnp.tile(Ivir, (Nt, 1, 1)) + jnp.einsum('nik, njk -> nij', tmats, tmats.conj())

    Uocc = jnp.linalg.cholesky(Mocc)
    Uocc_inv = jnp.linalg.inv(Uocc)
    Uocc_inv = jnp.einsum('nij -> nji', Uocc_inv).conj() 

    Uvir = jnp.linalg.cholesky(Mvir)
    Uvir_inv = jnp.linalg.inv(Uvir)
    Uvir_inv = jnp.einsum('nij -> nji', Uvir_inv).conj()

    TL = Uocc_inv
    BL = jnp.einsum('nik, nkj -> nij', tmats, Uocc_inv)
    TR = -jnp.einsum('nki, nkj -> nij', tmats.conj(), Uvir_inv)
    BR = Uvir_inv

    Top = jnp.concatenate([TL, TR], axis=2)
    Bot = jnp.concatenate([BL, BR], axis=2)
    U_tot = jnp.concatenate([Top, Bot], axis=1)
    
    return U_tot

def orthonormal_mos(tvecs):
    ta, tb = tvecs
    Ua = orthonormal_mos_spin(ta)
    Ub = orthonormal_mos_spin(tb)
    return (Ua, Ub)
