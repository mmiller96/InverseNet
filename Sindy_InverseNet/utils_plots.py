import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
from scipy.integrate import odeint

from loader import Loader
from evaluator import initialize_sindy_library_single
from example_lorenz import get_lorenz_data, generate_lorenz_data

def load_result_list(file_list, path_folder):
    result_list = []
    for file in file_list:
        file_name = path_folder + '/' + file
        with open(file_name, 'rb') as fp:
            results, params, coeff_mask, hyper_params = pickle.load(fp)
            result_list.append(results)
    return result_list

def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        results, params, coeff_mask, hyper_params = pickle.load(fp)
    return results, params, coeff_mask, hyper_params

def get_data(n_ics, dist='train'):
    t = np.arange(0, 5, .02)
    input_dim = 128
    # training data
    if dist == 'train':
        ic_means = np.array([0,0,25])
        ic_widths = 2*np.array([36,48,41])
        ics = ic_widths*(np.random.rand(n_ics, 3)-.5) + ic_means
    elif dist == 'test':
        inDist_ic_widths = np.array([36,48,41])
        outDist_extra_width = np.array([18,24,20])
        full_width = inDist_ic_widths + outDist_extra_width
        i = 0
        ics = np.zeros((n_ics,3))
        while i < n_ics:
            ic = np.array([np.random.uniform(-full_width[0],full_width[0]),
                           np.random.uniform(-full_width[1],full_width[1]),
                           np.random.uniform(-full_width[2],full_width[2]) + 25])
            if ((ic[0] > -inDist_ic_widths[0]) and (ic[0] < inDist_ic_widths[0])) \
                and ((ic[1] > -inDist_ic_widths[1]) and (ic[1] < inDist_ic_widths[1])) \
                and ((ic[2] > 25-inDist_ic_widths[2]) and (ic[2] < 25+inDist_ic_widths[2])):
                continue
            else:
                ics[i] = ic
                i += 1
    data = generate_lorenz_data(ics, t, input_dim, linear=False, normalization=np.array([1/40,1/40,1/40]))
    return data


def get_simulation_params(file_name, steps_inner=None, alpha=None):
    results, params, coeff_mask, hyper_params = load_pickle(file_name)
    print('model_name:  ' + str(hyper_params['model_name']))
    if alpha is not None:
        hyper_params['alpha'] = alpha
    if steps_inner is not None:
        print('steps_inner: ' + str(steps_inner) + '  trained with: ' + str(hyper_params['steps_inner']))
        hyper_params['steps_inner'] = steps_inner
    else:
        print('steps_inner: ' + str(hyper_params['steps_inner']))
    print('alpha:       ' + str(hyper_params['alpha']))
    rng = jax.random.PRNGKey(-1)
    loader = Loader(hyper_params)
    model, _ = loader.get_model_and_hyper_params(rng)

    if (hyper_params['model_name'] == 'IV'):
        params_psi = params[:hyper_params['n_psi']]
        params_phi = None
    else:
        params_psi = params[hyper_params['n_phi']:-1]
        params_phi = params[:hyper_params['n_phi']]
    sindy_coeff = params[-1][0]
    psi_vec = model.func['psi_vec']
    phi_vec = model.func['phi_vec']
    return phi_vec, psi_vec, coeff_mask, sindy_coeff, hyper_params, params_psi, params_phi

def simulate(hyper_params, coeff_mask, sindy_coeff, z_start, t):
    big_eps = np.multiply(coeff_mask, sindy_coeff)
    sindy_library = initialize_sindy_library_single(hyper_params)
    def f(z, t):
        return np.dot(sindy_library(z), big_eps)
    z_pred = odeint(f, z_start, t, printmessg=False)
    return z_pred

def calc_simulation_loss(phi_vec, psi_vec, coeff_mask, sindy_coeff, hyper_params, params_psi, params_phi, data):
    z_sim_model, z_rec_model = [], []
    x_sim_model, x_rec_model, x_model = [], [], []
    for x, z in zip(data['x'], data['z']):
        if hyper_params['model_name'] == 'IV':
            z0 = np.zeros((x.shape[0], hyper_params['z_latent']))
            z_rec = phi_vec(params_psi, x, z0)
        else:
            z_rec = phi_vec(params_phi, x)
        x_rec = psi_vec(params_psi, z_rec)
        z_sim = simulate(hyper_params, coeff_mask, sindy_coeff, z_rec[0], data['t'])
        x_sim = psi_vec(params_psi, z_sim)
        z_sim_model.append(z_sim)
        z_rec_model.append(z_rec)
        x_sim_model.append(x_sim)
        x_rec_model.append(x_rec)
        x_model.append(x)
    x_rec_model = np.vstack(x_rec_model)
    x_sim_model = np.vstack(x_sim_model)
    x_model = np.vstack(x_model)
    loss_rec = np.mean((x_model - x_rec_model) ** 2)
    loss_sim = np.mean((x_model - x_sim_model) ** 2)
    x_rec_diff = np.nan_to_num((x_model - x_rec_model)** 2, nan=100.0)
    x_sim_diff = np.nan_to_num((x_model - x_sim_model)** 2, nan=100.0)
    loss_rec_median = np.median(x_rec_diff)
    loss_sim_median = np.median(x_sim_diff)
    z_sim_model = np.array(z_sim_model)
    z_rec_model = np.array(z_rec_model)
    return loss_rec, loss_sim, loss_rec_median, loss_sim_median, z_rec_model, z_sim_model

