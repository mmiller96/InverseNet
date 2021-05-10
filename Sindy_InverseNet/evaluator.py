import numpy as np
from jax import jit
import jax.numpy as jnp
from utils_functions import reshape_and_to_numpy
from utils import display_multiple_img
import pdb
from scipy.integrate import odeint
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

class Evaluator():
    def __init__(self, X, dX, X_eval, dX_eval, hyper_params, func, t):
        """ This class calculates different error terms for training and evaluation set according to
            "Model-Order Reduction as an Inverse Problem".
            Additionally, SINDy is used to predict z for every 1s. For a simulation that goes for 5s, z is calculated
            from x for every second (0s, 1s, 2s, 3s, 4s). Each z is then simulated for 1s and combined to one
            big simulation z_pred. The L2-norm is calculated from z_pred and from the encoded z. The simulation goes
            only for 1s only since the trajectories are divergiong fast, for longer time periods. The error would not be
            meaningful anymore.

           Args:
                X (numpy.array): Training observations.
                dX (numpy.array): Derivatives of training observations.
                X_eval (numpy.array): Evaluation observations.
                dX_eval (numpy.array): Derivatives of evaluation observations.
                hyper_params (dict): hyperparameters, contains simulation settings.
                func (dict): Contains functions from the model.
                t (numpy.array): Time points. Needed for the simulations

       """
        self.X_sim, self.dX_sim, self.X_eval_sim, self.dX_eval_sim = reshape_and_to_numpy([X, dX, X_eval, dX_eval], hyper_params['train_num_simulations'], t.shape[0])
        self.X = X
        self.dX = dX
        self.X_eval = X_eval
        self.dX_eval = dX_eval
        self.func = func
        self.hyper_params = hyper_params
        self.t = t
        self.t1sec = t[t < 1.0]
        self.num_samples_per_1s = self.t1sec.shape[0]
        self.num_samples_per_simulation = t.shape[0]
        self.num_samples = hyper_params['num_simulations'] * t.shape[0]
        self.num_samples_train = hyper_params['train_num_simulations'] * t.shape[0]
        self.get_results_all = initialize_get_results_all(self)
        self.sindy_library_single = initialize_sindy_library_single(self.hyper_params)

    def evaluate(self, params_all, params_psi, coeff_mask):
        """ Calculates results for reconstrucion looses and simulation losses.

           Args:
                params_all (list): list of all parameters from a model. Last entry contains the sindy_library
                params_psi (list): forward function / decoding part of the model.
                coeff_mask (numpy.array): Mask for setting values below a threshold to zero.

            Return:
                results_train (dict): Training results for every "step_sample"
                results_eval (dict): Evaluation results for every "step_sample"
                results_train_sim (dict): Training results for the first "train_num_simulations"
                results_eval_sim (dict): Evaluation results for the first "num_simulations"
                z_pred_train (numpy.array): Simulated z in training set.
                z_pred_eval (numpy.array): Simulated z in evaluation set.
                x_pred_train (numpy.array): Simulated x in training set.
                x_pred_eval (numpy.array): Simulated x in evaluation set.

       """
        results_train, results_eval, results_train_sim, results_eval_sim = self.get_results_all(params_all, coeff_mask)
        z_real_train_1s, z_real_eval_1s = self.get_1s_z(results_train_sim, self.hyper_params['train_num_simulations']), self.get_1s_z(results_eval_sim, self.hyper_params['num_simulations'])
        z_pred_train, z_pred_eval = self.simulate(params_all, coeff_mask, z_real_train_1s, z_real_eval_1s)
        x_pred_train = self.func['psi_vec'](params_psi, z_pred_train)
        x_pred_eval = self.func['psi_vec'](params_psi, z_pred_eval)
        return results_train, results_eval, results_train_sim, results_eval_sim, z_pred_train, z_pred_eval, x_pred_train, x_pred_eval

    def simulate(self, params_all, coeff_mask, z_real_train_1s, z_real_eval_1s):
        """ Simualte z for every second.

                   Args:
                        params_all (list): list of all parameters from a model. Last entry contains the sindy_library
                        coeff_mask (numpy.array): Mask for setting values below a threshold to zero.
                        z_real_train_1s (numpy.array): Contains z for every second in training set.
                                         shape (number simulations, number z in a simulation, number features in z)
                        z_real_eval_1s (numpy.array): Contains z for every second in evaluation set.
                                         shape (number simulations, number z in a simulation, number features in z)

                    Return:
                        z_pred_train (numpy.array): Simulated z in training set.
                        z_pred_eval  (numpy.array): Simulated z in evaluation set.


               """
        sindy_coeff = params_all[-1][0]
        big_eps = np.multiply(coeff_mask, sindy_coeff)
        sindy_library_single = self.sindy_library_single
        def f(z, t):
            return np.dot(sindy_library_single(z), big_eps)
        t1sec = self.t1sec
        z_pred_train = Parallel(n_jobs=-1)(delayed(simulate_1_example)(z, f, t1sec) for z in z_real_train_1s)
        z_pred_eval = Parallel(n_jobs=-1)(delayed(simulate_1_example)(z, f, t1sec) for z in z_real_eval_1s)
        z_pred_train = np.array(z_pred_train).reshape((-1, self.hyper_params['z_latent']))
        z_pred_eval = np.array(z_pred_eval).reshape((-1, self.hyper_params['z_latent']))
        return z_pred_train, z_pred_eval

    def get_1s_z(self, results, num_simulations):
        """ Calculates results for reconstrucion looses and simulation losses.

           Args:
                results (dict): Results of simulations
                num_simulations (int): Number of simulations

            Return:
                z_1s (numpy.array): Contains z for every second. shape (number simulations, number z in a simulation, number features in z)

       """
        z_1s = results['z'][::self.num_samples_per_1s].reshape((num_simulations, -1, self.hyper_params['z_latent']))
        z_1s = np.asarray(z_1s)
        return z_1s

    def calculate_losses(self, results, index, save_img=False):
        """ Calculates all losses.

           Args:
                results (dict): Reconstruction and simulation results, more details in evauluate.
                index (int): Current epoch
                save_img (bool): Saves images in "picture/" folder.

            Return:
                losses (dict): All losses in training and evaluation for reconstruction and simulation.

       """
        results_train, results_eval, results_train_sim, results_eval_sim, z_pred_train, z_pred_eval, x_pred_train, x_pred_eval = results
        loss_sim_train_z, loss_sim_train_x = self.calculate_sim_loss(self.X, results_train_sim, z_pred_train, x_pred_train)
        loss_sim_eval_z, loss_sim_eval_x = self.calculate_sim_loss(self.X_eval, results_eval_sim, z_pred_eval, x_pred_eval)
        losses = {}
        losses = self.add_losses(losses, results_train, 'train')
        losses['train_sim_z'] = loss_sim_train_z
        losses['train_sim_x'] = loss_sim_train_x

        losses = self.add_losses(losses, results_eval, 'val')
        losses['val_sim_z'] = loss_sim_eval_z
        losses['val_sim_x'] = loss_sim_eval_x

        if(save_img):
            images = {'x0: ': [self.X_sim[0, :, 0], results_train_sim['x_rec'][:self.num_samples_per_simulation, 0]],
                      'x1: ': [self.X_sim[0, :, 1], results_train_sim['x_rec'][:self.num_samples_per_simulation, 1]],
                      'dx0: ': [self.dX_sim[0, :, 0], results_train_sim['dx_rec'][:self.num_samples_per_simulation, 0]],
                      'dx1: ': [self.dX_sim[0, :, 1], results_train_sim['dx_rec'][:self.num_samples_per_simulation, 1]],
                      'z0_sim: ': [results_train_sim['z'][:self.num_samples_per_simulation, 0], z_pred_train[:self.num_samples_per_simulation, 0]],
                      'z1_sim: ': [results_train_sim['z'][:self.num_samples_per_simulation, 1], z_pred_train[:self.num_samples_per_simulation, 1]],
                      'x0_sim: ': [self.X_sim[0, :, 0], x_pred_train[:self.num_samples_per_simulation, 0]],
                      'x1_sim: ': [self.X_sim[0, :, 1], x_pred_train[:self.num_samples_per_simulation, 1]],
                      'dz0: ': [results_train_sim['dz'][:self.num_samples_per_simulation, 0], results_train_sim['dz_pred'][:self.num_samples_per_simulation, 0]],
                      }
            display_multiple_img(images, index, 'X_rec', rows=3, cols=3)
        return losses


    def calculate_sim_loss(self, X, results_sim, z_pred, x_pred):
        """ Calculates the L2-norm of the encoded z and the simulated z.
            Calculates the L2-norm (median) of the decoded x and the simulated x. Nan values are set to a high number.

           Args:
               X (numpy.array): Real observations.
                results_sim (dict): Results for the first "num_simulations"
                z_pred (numpy.array): Simulated z.
                x_pred (numpy.array): Simulated x.

            Return:
                loss_sim_z (float): loss in z.
                loss_sim_x (float): loss in x.
               """
        loss_sim_z = np.nanmean((results_sim['z'] - z_pred) ** 2) / np.nanmean((results_sim['z'][1:] - results_sim['z'][:-1]) ** 2)
        x_diff = (X[:self.num_samples] - x_pred) ** 2
        x_diff = np.nan_to_num(x_diff, nan=1000)                # set nan to high numbers
        loss_sim_x = np.nanmedian(x_diff)
        return loss_sim_z, loss_sim_x

    def add_losses(self, losses, results, name):
        losses[name + '_x_loss'] = float(results['x_loss'])
        losses[name + '_dx_loss'] = float(results['dx_loss'])
        losses[name + '_dz_loss'] = float(results['dz_loss'])
        losses[name + '_regul'] = float(results['regul'])
        losses[name + '_loss'] = float(results['loss'])
        return losses

    def plot_losses(self, losses, index):
        self.plot_set(losses, index, name='train')
        self.plot_set(losses, index, name='val')
        print()

    def plot_set(self, losses, index, name='train'):
        print(name + ' ' +str(index) + ':   T_loss: ' + str(losses[str(name) + '_loss']) + '  T_recon: ' + str(losses[str(name) + '_x_loss']) +
              '  T_dz: ' + str(losses[str(name) + '_dz_loss']) + '  T_dx: ' + str(losses[str(name) + '_dx_loss']) + '  Regul: ' + str(losses[str(name) + '_regul']) +
              '   x_sim: ' + str(losses[str(name) + '_sim_x']) + '   z_sim: ' + str(losses[str(name) + '_sim_z']))

def initialize_get_results_all(self):
    """
    jit-Compiled functions for fast evaluations. The hyperparameters, the model and the trainings/evaluation set
    are part of the function "get_results_all" and are static. Only params_all and coeff_mask are variables.
    It is defined for the Autoencoder and the InverseNet.
    Every "step_sample" the result is calculated for training and evaluation.
    results_train and results_eval will be later used for calculating the reconstruction errors.
    results_train_sim and results_eval_sim contain the results of the first simulations in the trainingset and
    evaluationset. With "train_num_simulations" the number of simulations in the trainingset and with
    "num_simulations" the number of simulation in the evaluationset are defined. Reults for whole simulations are
    calculated and will be later used to compare the reconstructed X and z with the SINDy-model predicted X and z.

    Return:
        get_results_all (function): jit-compiled function for evaluating the reconstruction results and
                                    the simulation results.

    """
    if(self.hyper_params['model_name'] == 'IV'):
        @jit
        def get_results_all(params_all, coeff_mask):
            z0_train = jnp.zeros((self.X[::self.hyper_params['step_sample']].shape[0], self.hyper_params['z_latent']))
            z0_eval = jnp.zeros((self.X_eval[::self.hyper_params['step_sample']].shape[0], self.hyper_params['z_latent']))
            results_train = self.func['T_results'](params_all, self.X[::self.hyper_params['step_sample']], self.dX[::self.hyper_params['step_sample']], coeff_mask, z0_train)
            results_eval = self.func['T_results'](params_all, self.X_eval[::self.hyper_params['step_sample']], self.dX_eval[::self.hyper_params['step_sample']], coeff_mask, z0_eval)

            z0_sim_train = jnp.zeros((self.X[:self.num_samples_train].shape[0], self.hyper_params['z_latent']))
            z0_sim_eval = jnp.zeros((self.X_eval[:self.num_samples].shape[0], self.hyper_params['z_latent']))
            results_train_sim = self.func['T_results'](params_all, self.X[:self.num_samples_train], self.dX[:self.num_samples], coeff_mask, z0_sim_train)
            results_eval_sim = self.func['T_results'](params_all, self.X_eval[:self.num_samples], self.dX_eval[:self.num_samples], coeff_mask, z0_sim_eval)
            return results_train, results_eval, results_train_sim, results_eval_sim
    else:
        @jit
        def get_results_all(params_all, coeff_mask):
            results_train = self.func['T_results'](params_all, self.X[::self.hyper_params['step_sample']], self.dX[::self.hyper_params['step_sample']], coeff_mask)
            results_eval = self.func['T_results'](params_all, self.X_eval[::self.hyper_params['step_sample']], self.dX_eval[::self.hyper_params['step_sample']], coeff_mask)

            results_train_sim = self.func['T_results'](params_all, self.X[:self.num_samples_train], self.dX[:self.num_samples], coeff_mask)
            results_eval_sim = self.func['T_results'](params_all, self.X_eval[:self.num_samples], self.dX_eval[:self.num_samples], coeff_mask)
            return results_train, results_eval, results_train_sim, results_eval_sim
    return get_results_all

def initialize_sindy_library_single(hyper_params):
    """ Creates a function that calculates for a single latent variable z, candidate values.
        This function will be used for predicting the next latent variables.
        It is based on numpy so multi-core processing is possible.

        Args:
          hyper_params (dict): hyperparameters

        Returns:
          sindy_library_single (function): Function to create candidate functions.
    """
    def sindy_library_single(z):
        library = [1.0]
        for i in range(hyper_params['z_latent']):
            library.append(z[i])
        if hyper_params['poly_order'] > 1:
            for i in range(hyper_params['z_latent']):
                for j in range(i, hyper_params['z_latent']):
                    library.append(np.multiply(z[i], z[j]))
        if hyper_params['poly_order'] > 2:
            for i in range(hyper_params['z_latent']):
                for j in range(i, hyper_params['z_latent']):
                    for k in range(j, hyper_params['z_latent']):
                        z1 = np.multiply(z[i], z[j])
                        library.append(np.multiply(z1, z[k]))
        if hyper_params['include_sine']:
            for i in range(hyper_params['z_latent']):
                library.append(np.sin(z[i]))
        return np.hstack(library)
    return sindy_library_single

def simulate_1_example(z_real, f, t1sec):
    """ predict z for a simulation using lsoda from the FORTRAN library odepack.

            Args:
                z_real: z sampled every second in a simulation.
                t1sec (numpy.array): Time points for 1s.
                f (function): Derivative of a function. Is used for integration.
            Returns:
              numpy.array: Predicted z for a whole simulation
        """
    z_pred = []
    for i, z_real_1sec in enumerate(z_real):
        z_pred.append(odeint(f, z_real_1sec, t1sec, printmessg=False))
    return np.vstack(z_pred)