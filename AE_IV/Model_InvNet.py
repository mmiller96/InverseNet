from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap, jit, jacrev
from jax.experimental.optimizers import adam
import pdb

from Utils import save_obj, display_multiple_img
from Utils_Functions import serial


def initialize_functions(hyper_params, psi, g, get_params_from_opt, opt_update):
    psi_z = jacrev(psi, argnums=1)
    psi = jit(psi)

    g = jit(g)

    def L(params_psi, x, z, z_ref):
        return jnp.sum((x - psi(params_psi, z, z_ref)) ** 2)

    L_z = grad(L, argnums=2)

    def big_phi(params_psi, x, z_ref, z, _):
        return z - hyper_params['alpha'] * L_z(params_psi, x, z, z_ref), z

    @jit
    def phi(params_psi, x, z, z_ref):
        big_phi_fn = partial(big_phi, params_psi, x, z_ref)
        z, _ = jax.lax.scan(big_phi_fn, z, jnp.zeros(hyper_params['steps_inner']), length=hyper_params['steps_inner'])
        return z

    phi_x = jacrev(phi, argnums=1)

    def get_all_L(params_psi, x, z, z_ref):
        big_phi_fn = partial(big_phi, params_psi, x, z_ref)
        z, z_all = jax.lax.scan(big_phi_fn, z, jnp.zeros(hyper_params['steps_inner']),
                                length=hyper_params['steps_inner'])
        L_all = jnp.array([L(params_psi, x, z, z_ref) for z in z_all])
        return L_all

    get_all_L_vmap = jit(vmap(get_all_L, (None, 0, 0, 0)))

    def l1_regularization(params, penalty):
        val = 0.0
        for layer in params:
            for param in layer:
                val += jnp.sum(jnp.abs(param))
        return val * penalty

    def l2_regularization(params, penalty):
        val = 0.0
        for layer in params:
            for param in layer:
                val += jnp.sum(param ** 2)
        return val * penalty

    def T(params_all, x, dx, z, z_ref):
        params_psi, params_g = params_all[:hyper_params['n_psi']], params_all[hyper_params['n_psi']:]
        z_opt = phi(params_psi, x, z, z_ref)
        dz_pred = g(params_g, z_opt, z_ref)
        dx_recon = jnp.dot(psi_z(params_psi, z_opt, z_ref), dz_pred)
        z_opt_x = phi_x(params_psi, x, z, z_ref)

        x_loss = jnp.sum((x - psi(params_psi, z_opt, z_ref)) ** 2)
        dx_loss = hyper_params['eta1'] * jnp.sum((dx - dx_recon) ** 2)
        dz_loss = hyper_params['eta2'] * jnp.sum((jnp.dot(z_opt_x, dx) - dz_pred) ** 2)
        regul = l1_regularization(params_g, hyper_params['eta3'])

        loss = x_loss + dx_loss + dz_loss + regul
        return loss

    def T_seperate(params_all, x, dx, z, z_ref):
        params_psi, params_g = params_all[:hyper_params['n_psi']], params_all[hyper_params['n_psi']:]
        z_opt = phi(params_psi, x, z, z_ref)
        dz_pred = g(params_g, z_opt, z_ref)
        dx_recon = jnp.dot(psi_z(params_psi, z_opt, z_ref), dz_pred)
        z_opt_x = phi_x(params_psi, x, z, z_ref)

        x_loss = jnp.sum((x - psi(params_psi, z_opt, z_ref)) ** 2)
        dx_loss = hyper_params['eta1'] * jnp.sum((dx - dx_recon) ** 2)
        dz_loss = hyper_params['eta2'] * jnp.sum((jnp.dot(z_opt_x, dx) - dz_pred) ** 2)
        regul = l1_regularization(params_g, hyper_params['eta3'])

        loss = x_loss + dx_loss + dz_loss + regul
        return loss, x_loss, dx_loss, dz_loss, dx_recon, regul

    T_params = grad(T, argnums=0)

    # vectorized functions
    psi_vec = jit(vmap(psi, (None, 0, 0)))
    phi_vec = jit(vmap(phi, (None, 0, 0, 0)))
    T_seperate_vec = jit(vmap(T_seperate, (None, 0, 0, 0, 0)))
    T_params_vec = jit(vmap(T_params, (None, 0, 0, 0, 0)))

    @jit
    def update(i, opt_state, x, dx, z_ref):
        #key = jax.random.PRNGKey(i)
        #z = jax.random.normal(key, (hyper_params['batch_size'], hyper_params['z_latent']))
        z = jnp.zeros((hyper_params['batch_size'], hyper_params['z_latent']))
        params = get_params_from_opt(opt_state)
        grads = T_params_vec(params, x, dx, z, z_ref)
        grads_mean = []
        for i, layer in enumerate(grads):
            if (len(layer) == 0):
                grads_mean.append(())
            else:
                grads_mean.append(tuple([jnp.mean(weight, axis=0) for weight in layer]))
        return opt_update(i, grads_mean, opt_state)

    return update, T_seperate_vec, phi_vec, psi_vec, phi, get_all_L_vmap, L_z, T, L, T_params, get_all_L, g


def get_model_and_optimizer(model_params, hyper_params, rng):
    init_fun, psi, g = serial(model_params)  # load model from serial
    _, init_params = init_fun(rng, (hyper_params['batch_size'], hyper_params['z_latent'] + 4))  # get initial params
    opt_init, opt_update, get_params_from_opt = adam(hyper_params['lr'])  # get optimizer
    opt_state = opt_init(init_params)
    return psi, g, opt_update, opt_state, get_params_from_opt, init_params, opt_init


class InvNet():
    def __init__(self, model_params, hyper_params, rng):
        self.psi, g, opt_update, self.opt_state, self.get_params_from_opt, init_params, self.opt_init = get_model_and_optimizer(
            model_params, hyper_params, rng)
        self.hyper_params = hyper_params
        self.update, self.T_seperate_vec, self.phi_vec, self.psi_vec, self.phi, self.get_L, self.L_z, self.T, self.L, self.T_params, self.get_all_L, self.g = initialize_functions(
            hyper_params, self.psi, g, self.get_params_from_opt, opt_update)

    def forward(self, i, x, dx, z_ref):
        self.opt_state = self.update(i, self.opt_state, x, dx, z_ref)

    def evaluate(self, X_all, dX_all, z_ref_all, step_sample, k, name=None):
        X, dX, z_ref = X_all[::step_sample], dX_all[::step_sample], z_ref_all[::step_sample]
        z = jnp.zeros((X.shape[0], self.hyper_params['z_latent']))
        params_all = self.get_params()
        params_psi, params_g = params_all[:self.hyper_params['n_psi']], params_all[self.hyper_params['n_psi']:]
        T = self.T_seperate_vec(params_all, X, dX, z, z_ref)
        loss_sim, z_opt, z_pred = self.predict_45s(X_all, z_ref_all)
        L = self.get_L(params_psi, X, z, z_ref)
        L = jnp.mean(L, axis=0)
        x_rec = self.psi_vec(params_psi, z_opt[::step_sample], z_ref)  # .reshape(-1, 400)
        dx_rec = T[4]
        regul = T[5]
        images = {'P_I: ': [X[:, 0], x_rec[:, 0]], 'Q_I: ': [X[:, 1], x_rec[:, 1]], 'dP_I: ': [dX[:, 0], dx_rec[:, 0]],
                  'dQ_I: ': [dX[:, 0], dx_rec[:, 1]],
                  'z0: ': [z_opt[::step_sample, 0], z_pred[::step_sample, 0]], 'z1: ': [z_opt[::step_sample, 1], z_pred[::step_sample, 1]],
                  'x_random1: ': [X[:, 16], x_rec[:, 16]], 'dx_random1: ': [dX[:, 16], dx_rec[:, 16]],
                  'L': L}
        if (name is None): name = 'X_rec'
        display_multiple_img(images, k, name, rows=3, cols=3)
        print(str(k) + ':   T_loss: ' + str(jnp.mean(T[0])) + '  T_recon: ' + str(jnp.mean(T[1])) + '  T_dx: ' + str(
            jnp.mean(T[2])) + '  z_dx: ' + str(jnp.mean(T[3])) + '  Regul: ' + str(jnp.mean(regul)) + '  loss_sim: ' + str(loss_sim))



    def predict_45s(self, x, z_ref):
        params_all = self.get_params()
        params_psi, params_g = params_all[:self.hyper_params['n_psi']], params_all[self.hyper_params['n_psi']:]
        z0 = jnp.zeros((len(x), self.hyper_params['z_latent']))
        z_pred = np.zeros((len(x), self.hyper_params['z_latent']))
        z_real = self.phi_vec(params_psi, x, z0, z_ref)
        for i, (z_ref_next, z) in enumerate(zip(z_ref, z_real)):
            dz = self.g(params_g, z, z_ref_next)
            z_pred[i, :] = z + dz
        z_opt = z_real
        z_real, z_pred, z_interpol = z_real[1:, :], z_pred[:-1, :], z_real[:-1, :]
        loss_ratio = jnp.sum((z_real - z_pred)**2)/jnp.sum((z_real - z_interpol)**2)
        return loss_ratio, z_opt, z_real


    def get_params(self):
        return self.get_params_from_opt(self.opt_state)

    def save(self, k):
        name = 'models/' + str(k)
        save_obj(self.get_params(), name)
