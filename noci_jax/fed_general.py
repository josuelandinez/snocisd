import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from noci_jax import slater_general

def optimize_fed(mf, h1e, h2e, e_nuc, fixed_tvecs=None, steps=300, lr=0.02):
    """
    Greedy sequential FED optimization. Finds the NEXT best non-orthogonal state
    while keeping the previously discovered states (fixed_tvecs) frozen.
    """
    nocc_a, nocc_b = mf.nelec
    nvir_a = mf.mo_coeff[0].shape[1] - nocc_a
    nvir_b = mf.mo_coeff[1].shape[1] - nocc_b

    if fixed_tvecs is None:
        ta_fixed = jnp.zeros((1, nvir_a, nocc_a))
        tb_fixed = jnp.zeros((1, nvir_b, nocc_b))
    else:
        ta_fixed, tb_fixed = fixed_tvecs

    # Initialize the ONE new state with random noise
    key = jax.random.PRNGKey(np.random.randint(10000))
    key_a, key_b = jax.random.split(key)
    t_a = jax.random.normal(key_a, (1, nvir_a, nocc_a)) * 0.1
    t_b = jax.random.normal(key_b, (1, nvir_b, nocc_b)) * 0.1
    params = (t_a, t_b)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def cost_fn(t_params):
        ta_all = jnp.concatenate([ta_fixed, t_params[0]], axis=0)
        tb_all = jnp.concatenate([tb_fixed, t_params[1]], axis=0)
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
            print(f"      FED Step {i:>3}: Energy = {loss:.8f} Ha")

    return params
