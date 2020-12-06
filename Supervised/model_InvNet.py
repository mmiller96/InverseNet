import jax
import jax.numpy as jnp
from jax.nn import relu
from jax import grad, vmap, jit
from jax.experimental.optimizers import sgd, adam
import pdb
import torch
import numpy as np
from functools import partial

def initialize_functions(hype_params, decode, f_sim, nlayers_decode, get_params, opt_update, z_latent, etha):
    decode_vec = jit(vmap(decode, (None, 0)))
    f_sim_vec = jit(vmap(f_sim, (None, 0)))
    def L(params, x, z):
        return jnp.sum((decode(params, z) - x)**2)

    dL_dz = grad(L, argnums=2)   # (params, x, z)
    L_vec = vmap(L, (None, 0, 0))

    def encode_without_hp(hype_params, params, x, z):
        alpha, steps_inner = hype_params
        for i in range(steps_inner):
            z = z - alpha * dL_dz(params, x, z)
        return z

    encode = partial(encode_without_hp, hype_params)
    encode_vec = jit(vmap(encode, (None, 0, 0)))
    def T(params, x, y, z):
        params, params_sim = params[:nlayers_decode], params[nlayers_decode:]
        z_opt = encode(params, x, z)
        x_opt = decode(params, z_opt)
        y_pred = f_sim(params_sim, z_opt)
        recon_loss = jnp.sum((x - x_opt)**2)
        classify_loss = jnp.sum((y - y_pred)**2)
        return recon_loss + classify_loss * etha
    T_vec = jit(vmap(T, (None, 0, 0, 0)))
    dT_dparam = grad(T, argnums=0)

    def T_classify(params, x, y, z):
        params, params_sim = params[:nlayers_decode], params[nlayers_decode:]
        z_opt = encode(params, x, z)
        y_pred = f_sim(params_sim, z_opt)
        return jnp.sum((y - y_pred)**2)
    T_classify_vec = jit(vmap(T_classify, (None, 0, 0, 0)))
    @jit
    def update(i, opt_state, x, y):
        z = jnp.zeros((x.shape[0], z_latent))
        params = get_params(opt_state)
        grads = dT_dparam(params, x, y, z)
        return opt_update(i, grads, opt_state)

    return encode_vec, decode_vec, L_vec, T_vec, update, T_classify_vec, f_sim_vec

class invNet_jax():
    def __init__(self, init_params, decode, f_sim, nlayers_decode, alpha, etha, steps_inner, z_latent, lr):
        self.nlayers_decode = nlayers_decode
        self.z_latent = z_latent
        self.decode = decode
        self.hype_params = (alpha, steps_inner)
        opt_init, opt_update, self.get_params = adam(lr)
        self.opt_state = opt_init(init_params)
        self.encode_vec, self.decode_vec, self.L_vec, self.T_vec, self.update, self.T_classify_vec, self.f_sim_vec = initialize_functions(self.hype_params, decode, f_sim, nlayers_decode, self.get_params, opt_update, z_latent, etha)
        self.lr = lr

    def forward(self, i, x, y):
        self.opt_state = self.update(i, self.opt_state, x, y)
