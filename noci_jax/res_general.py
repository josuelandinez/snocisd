import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from noci_jax import slater_general

def optimize_reshf(mf, h1e, h2e, e_nuc, init_tvecs, steps=300, lr=0.01):
    """
    Simultaneous Resonating HF optimization. Polishes multiple non-orthogonal 
    states collectively to find the true global multi-reference minimum.
    """
    nocc_a, nocc_b = mf.nelec
    nvir_a = mf.mo_coeff[0].shape[1] - nocc_a
    nvir_b = mf.mo_coeff[1].shape[1] - nocc_b

    # Base HF state is always included and fixed at T=0
    ta_hf = jnp.zeros((1, nvir_a, nocc_a))
    tb_hf = jnp.zeros((1, nvir_b, nocc_b))

    # Parameters to optimize: The N extra states (seeded by FED)
    params = init_tvecs

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def cost_fn(t_params):
        ta_all = jnp.concatenate([ta_hf, t_params[0]], axis=0)
        tb_all = jnp.concatenate([tb_hf, t_params[1]], axis=0)
        rmats = slater_general.tvecs_to_rmats((ta_all, tb_all), (nvir_a, nvir_b), (nocc_a, nocc_b))
        
        smat, hmat = slater_general.build_noci_matrices(rmats, mf.mo_coeff, h1e, h2e, e_nuc)
        
        s_eig, s_vec = jnp.linalg.eigh(smat)
        inv_sqrt_s = jnp.where(s_eig > 1e-10, 1.0 / jnp.sqrt(s_eig), 0.0)
        X = s_vec @ jnp.diag(inv_sqrt_s)
        h_prime = X.T @ hmat @ X
        vals, _ = jnp.linalg.eigh(h_prime)
        return vals[0]

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(cost_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for i in range(steps):
        params, opt_state, loss = update(params, opt_state)
        if i % 50 == 0 or i == steps - 1:
            print(f"      ResHF Step {i:>3}: Energy = {loss:.8f} Ha")

    return params
