from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap, jit, jacrev
from jax.experimental.optimizers import adam
import pdb

from utils import save_obj, display_multiple_img
from utils_functions import serial_autoencoder, init_sindy_library

class Autoencoder():
    def __init__(self, model_params, hyper_params, rng):
        phi, psi, opt_update, self.opt_state, self.get_params_from_opt, init_params, self.opt_init = \
            get_model_and_optimizer(model_params, hyper_params, rng)
        self.hyper_params = hyper_params
        self.update, self.update_refinement, self.func = initialize_functions(hyper_params, phi, psi, self.get_params_from_opt, opt_update)
        self.coeff_mask = jnp.ones((self.hyper_params['library_dim'], self.hyper_params['z_latent']))
        self.model_name = 'AE'

    def forward(self, i, x, dx, start_refinement):
        if(start_refinement):
            self.opt_state = self.update_refinement(i, self.opt_state, x, dx, self.coeff_mask)
        else:
            self.opt_state = self.update(i, self.opt_state, x, dx, self.coeff_mask)

    def get_params(self):
        return self.get_params_from_opt(self.opt_state)
    def get_params_psi(self):
        return self.get_params_from_opt(self.opt_state)[self.hyper_params['n_phi']:-1]

    def save(self, k):
        name = 'models/' + str(k)
        save_obj(self.get_params(), name)

def get_model_and_optimizer(model_params, hyper_params, rng):
    init_fun, phi, psi = serial_autoencoder(model_params, hyper_params)  # load model from serial
    _, init_params = init_fun(rng, (hyper_params['batch_size'], hyper_params['x_dim']))  # get initial params
    opt_init, opt_update, get_params_from_opt = adam(hyper_params['lr'])  # get optimizer
    opt_state = opt_init(init_params)
    return phi, psi,  opt_update, opt_state, get_params_from_opt, init_params, opt_init

def initialize_functions(hyper_params, phi, psi, get_params_from_opt, opt_update):
    psi_vec = jit(vmap(psi, (None, 0)))
    phi_vec = jit(vmap(phi, (None, 0)))

    sindy_library = init_sindy_library(hyper_params)

    psi_dz = jacrev(psi, argnums=1)
    psi = jit(psi)
    phi = jit(phi)
    phi_x = jacrev(phi, argnums=1)
    def dx_network(psi_params, z, dz_pred):
        dx_rec_dz = psi_dz(psi_params, z)
        dx_rec = jnp.dot(dx_rec_dz, dz_pred)
        return dx_rec
    dx_network_vec = vmap(dx_network, (None, 0, 0))
    def dz_pred_func(Theta, coeff_mask, sindy_coeff):
        return jnp.dot(Theta, jnp.multiply(coeff_mask, sindy_coeff))
    def dz_func(params_phi, x, dx):
        z_opt_x = phi_x(params_phi, x)
        return jnp.dot(z_opt_x, dx)
    dz_func_vec = vmap(dz_func, (None, 0, 0))

    dz_pred_vec = vmap(dz_pred_func, (0, None, None))

    def T(params_all, x, dx, coeff_mask):
        params_phi = params_all[:hyper_params['n_phi']]
        params_psi = params_all[hyper_params['n_phi']:-1]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_phi, x)
        Theta = sindy_library(z_opt)
        dz = dz_func_vec(params_phi, x, dx)

        dz_pred = dz_pred_vec(Theta, coeff_mask, sindy_coeff)
        x_rec = psi_vec(params_psi, z_opt)
        dx_rec = dx_network_vec(params_psi, z_opt, dz_pred)

        x_loss = jnp.mean(jnp.power(x - x_rec, 2))
        dx_loss = jnp.multiply(hyper_params['eta1'], jnp.mean(jnp.power(dx - dx_rec, 2)))
        dz_loss = jnp.multiply(hyper_params['eta2'], jnp.mean(jnp.power(dz - dz_pred, 2)))
        regul = jnp.multiply(hyper_params['eta3'], jnp.mean(jnp.abs(sindy_coeff)))

        loss = x_loss + dx_loss + dz_loss + regul
        return loss

    def T_refinement(params_all, x, dx, coeff_mask):
        params_phi = params_all[:hyper_params['n_phi']]
        params_psi = params_all[hyper_params['n_phi']:-1]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_phi, x)
        Theta = sindy_library(z_opt)
        dz = dz_func_vec(params_phi, x, dx)

        dz_pred = dz_pred_vec(Theta, coeff_mask, sindy_coeff)
        x_rec = psi_vec(params_psi, z_opt)
        dx_rec = dx_network_vec(params_psi, z_opt, dz_pred)

        x_loss = jnp.mean(jnp.power(x - x_rec, 2))
        dx_loss = jnp.multiply(hyper_params['eta1'], jnp.mean(jnp.power(dx - dx_rec, 2)))
        dz_loss = jnp.multiply(hyper_params['eta2'], jnp.mean(jnp.power(dz - dz_pred, 2)))

        loss = x_loss + dx_loss + dz_loss
        return loss
    @jit
    def T_seperate(params_all, x, dx, coeff_mask):
        params_phi = params_all[:hyper_params['n_phi']]
        params_psi = params_all[hyper_params['n_phi']:-1]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_phi, x)
        Theta = sindy_library(z_opt)
        dz = dz_func_vec(params_phi, x, dx)

        dz_pred = dz_pred_vec(Theta, coeff_mask, sindy_coeff)
        x_rec = psi_vec(params_psi, z_opt)
        dx_rec = dx_network_vec(params_psi, z_opt, dz_pred)

        results = {}
        results['x_loss'] = jnp.mean(jnp.power(x - x_rec, 2))
        results['dx_loss'] = np.mean(jnp.power(dx - dx_rec, 2))
        results['dz_loss'] = jnp.mean(jnp.power(dz - dz_pred, 2))
        results['regul'] = jnp.mean(jnp.abs(sindy_coeff))

        dx_loss = jnp.multiply(hyper_params['eta1'], results['dx_loss'])
        dz_loss = jnp.multiply(hyper_params['eta2'], results['dz_loss'])
        regul = jnp.multiply(hyper_params['eta3'], results['regul'])

        results['loss'] = results['x_loss'] + dx_loss + dz_loss + regul
        results['x_rec'] = x_rec
        results['dx_rec'] = dx_rec
        results['z'] = z_opt
        results['dz'] = dz
        results['dz_pred'] = dz_pred
        return results

    T_params = grad(T, argnums=0)
    T_refinement_params = grad(T_refinement, argnums=0)

    @jit
    def update(i, opt_state, x, dx, coeff_mask):
        params = get_params_from_opt(opt_state)
        grads = T_params(params, x, dx, coeff_mask)
        return opt_update(i, grads, opt_state)

    @jit
    def update_refinement(i, opt_state, x, dx, coeff_mask):
        params = get_params_from_opt(opt_state)
        grads = T_refinement_params(params, x, dx, coeff_mask)
        return opt_update(i, grads, opt_state)

    func = {'psi': psi, 'psi_vec': psi_vec, 'phi': phi, 'phi_vec': phi_vec, 'T_results': T_seperate,
            'sindy_library': sindy_library,  'T': T}
    return update, update_refinement, func