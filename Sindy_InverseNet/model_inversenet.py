from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap, jit, jacrev, device_put
from jax.experimental.optimizers import sgd, rmsprop_momentum, adam
import pdb

from utils import save_obj, display_multiple_img
from utils_functions import serial_InvNet, init_sindy_library

class Inversenet():
    def __init__(self, model_params, hyper_params, rng):
        psi, opt_update, self.opt_state, self.get_params_from_opt, init_params, self.opt_init = \
            get_model_and_optimizer(model_params, hyper_params, rng)
        self.hyper_params = hyper_params
        self.update, self.update_refinement, self.func = initialize_functions(hyper_params, psi, self.get_params_from_opt, opt_update)
        self.coeff_mask = jnp.ones((self.hyper_params['library_dim'], self.hyper_params['z_latent']))

    def forward(self, i, x, dx, start_refinement):
        """ Trains the model for one batch.

           Args:
                i (int): Current gradient step. (needed for opt_update)
                x (numpy.array): observations (samples, features)
                dx (numpy.array): derivative of observations (samples, features)
                start_refinement (bool): If True, trains without regularization.

       """
        if(start_refinement):
            self.opt_state = self.update_refinement(i, self.opt_state, x, dx, self.coeff_mask)
        else:
            self.opt_state = self.update(i, self.opt_state, x, dx, self.coeff_mask)

    def get_params(self):
        return self.get_params_from_opt(self.opt_state)

    def get_params_psi(self):
        return self.get_params_from_opt(self.opt_state)[:self.hyper_params['n_psi']]


def get_model_and_optimizer(model_params, hyper_params, rng):
    """ Trains the model for one batch.

       Args:
            model_params (list): Contains layer and activation weights for the forward function.
            hyper_params (dict): Hyperparameters of the model.
            rng (jax.random.PRNGKey): Random key for initializing parameter weights

        Returns:
             psi (func): forward function
             opt_update (func): Function to update the state.
             opt_state (func): Function
             get_params_from_opt,
             init_params (list): Start parameters.
             opt_init()

   """
    init_fun, psi = serial_InvNet(model_params, hyper_params)  # load model from serial
    _, init_params = init_fun(rng, (hyper_params['batch_size'], hyper_params['z_latent']))  # get initial params
    opt_init, opt_update, get_params_from_opt = adam(hyper_params['lr'])  # get optimizer
    opt_state = opt_init(init_params)
    return psi, opt_update, opt_state, get_params_from_opt, init_params, opt_init

def initialize_functions(hyper_params, psi, get_params_from_opt, opt_update):
    psi = jit(psi)
    psi_vec = jit(vmap(psi, (None, 0)))

    def L(params_psi, x, z):
        return jnp.sum(jnp.power(x - psi(params_psi, z), 2))
    L_z = grad(L, argnums=2)
    def big_phi(params_psi, x, z, _):
        return z - hyper_params['alpha'] * L_z(params_psi, x, z), z
    def phi(params_psi, inputs, z):     # inputs = x
        # in jax.lax.scan, every step takes an input and an output from the last time point to calculate the next
        #  time point. An input has to be defined that's why zeros are going into the function. They are just a placeholders.
        big_phi_fn = partial(big_phi, params_psi, inputs)
        z, _ = jax.lax.scan(big_phi_fn, z, jnp.zeros(hyper_params['steps_inner']), length=hyper_params['steps_inner'])
        return z
    phi_vec = jit(vmap(phi, (None, 0, 0)))
    sindy_library = init_sindy_library(hyper_params)

    psi_dz = jacrev(psi, argnums=1)
    phi_x = jacrev(phi, argnums=1)

    def dx_network(psi_params, z, dz_pred):
        dx_rec_dz = psi_dz(psi_params, z)
        dx_rec = jnp.dot(dx_rec_dz, dz_pred)
        return dx_rec
    dx_network_vec = vmap(dx_network, (None, 0, 0))
    def dz_pred_func(Theta, coeff_mask, sindy_coeff):
        return jnp.dot(Theta, jnp.multiply(coeff_mask, sindy_coeff))
    def dz_func(params_psi, x, dx, z):
        z_opt_x = phi_x(params_psi, x, z)
        return jnp.dot(z_opt_x, dx)
    dz_func_vec = vmap(dz_func, (None, 0, 0, 0))

    dz_pred_vec = vmap(dz_pred_func, (0, None, None))

    def T(params_all, x, dx, coeff_mask, z0):
        params_psi = params_all[:hyper_params['n_psi']]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_psi, x, z0)
        Theta = sindy_library(z_opt)

        dz = dz_func_vec(params_psi, x, dx, z0)

        dz_pred = dz_pred_vec(Theta, coeff_mask, sindy_coeff)
        x_rec = psi_vec(params_psi, z_opt)
        dx_rec = dx_network_vec(params_psi, z_opt, dz_pred)

        x_loss = jnp.mean(jnp.power(x - x_rec, 2))
        dx_loss = jnp.multiply(hyper_params['eta1'], jnp.mean(jnp.power(dx - dx_rec, 2)))
        dz_loss = jnp.multiply(hyper_params['eta2'], jnp.mean(jnp.power(dz - dz_pred, 2)))
        regul = jnp.multiply(hyper_params['eta3'], jnp.mean(jnp.abs(sindy_coeff)))

        loss = x_loss + dx_loss + dz_loss + regul
        return loss

    def T_refinement(params_all, x, dx, coeff_mask, z0):
        params_psi = params_all[:hyper_params['n_psi']]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_psi, x, z0)
        Theta = sindy_library(z_opt)

        dz = dz_func_vec(params_psi, x, dx, z0)

        dz_pred = dz_pred_vec(Theta, coeff_mask, sindy_coeff)
        x_rec = psi_vec(params_psi, z_opt)
        dx_rec = dx_network_vec(params_psi, z_opt, dz_pred)

        x_loss = jnp.mean(jnp.power(x - x_rec, 2))
        dx_loss = jnp.multiply(hyper_params['eta1'], jnp.mean(jnp.power(dx - dx_rec, 2)))
        dz_loss = jnp.multiply(hyper_params['eta2'], jnp.mean(jnp.power(dz - dz_pred, 2)))

        loss = x_loss + dx_loss + dz_loss
        return loss

    @jit
    def T_seperate(params_all, x, dx, coeff_mask, z0):
        params_psi = params_all[:hyper_params['n_psi']]
        sindy_coeff = params_all[-1][0]

        z_opt = phi_vec(params_psi, x, z0)
        Theta = sindy_library(z_opt)

        dz = dz_func_vec(params_psi, x, dx, z0)

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
        z0 = jnp.zeros((hyper_params['batch_size'], hyper_params['z_latent']))
        grads = T_params(params, x, dx, coeff_mask, z0)
        return opt_update(i, grads, opt_state)

    @jit
    def update_refinement(i, opt_state, x, dx, coeff_mask):
        params = get_params_from_opt(opt_state)
        z0 = jnp.zeros((hyper_params['batch_size'], hyper_params['z_latent']))
        grads = T_refinement_params(params, x, dx, coeff_mask, z0)
        return opt_update(i, grads, opt_state)

    func = {'psi': psi, 'psi_vec': psi_vec, 'phi': phi, 'phi_vec': phi_vec, 'T_results': T_seperate,
            'sindy_library': sindy_library, 'T': T}
    return update, update_refinement, func