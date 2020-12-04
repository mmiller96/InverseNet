import jax
import jax.numpy as jnp
from jax.nn import relu
from jax import grad, vmap, jit
from jax.experimental.optimizers import sgd, adam
import pdb
import torch
import numpy as np
from functools import partial

def initialize_functions(hype_params, decode, get_params, opt_update, z_latent):
    decode_vec = jit(vmap(decode, (None, 0)))
    def L(params, x, z):
        return jnp.sum((decode(params, z) - x)**2)

    dL_dz = grad(L, argnums=2)   # (params, x, z)
    dL_dz_vec = vmap(dL_dz, (None, 0, 0))
    L_vec = vmap(L, (None, 0, 0))

    def encode_without_hp(hype_params, params, x, z):
        alpha, steps_inner = hype_params
        for i in range(steps_inner):
            z = z - alpha * dL_dz(params, x, z)
        return z

    encode = partial(encode_without_hp, hype_params)
    encode_vec = jit(vmap(encode, (None, 0, 0)))
    def T(params, x, z):
        z_opt = encode(params, x, z)
        x_opt = decode(params, z_opt)
        return jnp.sum((x - x_opt)**2)
    T_vec = jit(vmap(T, (None, 0, 0)))
    dT_dparam = grad(T, argnums=0)

    @jit
    def update(i, opt_state, x):
        z = jnp.zeros((x.shape[0], z_latent))
        params = get_params(opt_state)
        grads = dT_dparam(params, x, z)
        return opt_update(i, grads, opt_state)

    return encode_vec, decode_vec, L_vec, T_vec, update

class invNet_jax():
    def __init__(self, decode, init_params, alpha, steps_inner, z_latent, lr):
        self.z_latent = z_latent
        self.decode = decode
        self.hype_params = (alpha, steps_inner)
        #opt_init, opt_update, get_params = sgd(lr)
        opt_init, opt_update, get_params = adam(lr)
        self.opt_state = opt_init(init_params)
        self.get_params = get_params
        self.encode_vec, self.decode_vec, self.L_vec, self.T_vec, self.update = initialize_functions(self.hype_params, decode, get_params, opt_update, z_latent)
        self.lr = lr

    def forward(self, i, x):
        self.opt_state = self.update(i, self.opt_state, x)
