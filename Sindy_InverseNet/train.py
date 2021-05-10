import pdb
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
from utils import loss_dummy_model
from evaluator import Evaluator

class Trainer():
    def __init__(self, X, dX, X_eval, dX_eval, t, model, hyper_params):
        """
            Trains a model which is combined with SINDy.
            Together, proper latent variables and the partial differentail equations are found.
            More details in "Data-driven discovery of coordinates and governing equations"

            Arguments:
                X (numpy.array): Training observations
                dX (numpy.array): Derivatives of training observations
                X_eval (numpy.array): Evaluation observations
                dX_eval (numpy.array): Derivatives of evaluation observations
                t (numpy.array): Time points. Needed for the simulations
                hyper_params (dict): model hyper_params and settings for training

            """
        self.X = X
        self.dX = dX
        self.X_eval = X_eval
        self.dX_eval = dX_eval
        self.model = model
        self.evaluator = Evaluator(X, dX, X_eval, dX_eval, hyper_params, model.func, t)
        self.hyper_params = hyper_params
        self.start_refinement = False
        self.training_results = pd.DataFrame(columns=['train_x_loss', 'train_dx_loss', 'train_dz_loss', 'train_regul',
                                                      'train_loss', 'train_sim_z', 'train_sim_x', 'val_x_loss',
                                                      'val_dx_loss', 'val_dz_loss', 'val_regul', 'val_loss',
                                                      'val_sim_z', 'val_sim_x', 'epoch'])
        self.best_loss = 10000.0
        self.number_best = 0

    def fit(self, rng_batch):
        """
            Trains a model based on the hyperparameters.

            Arguments:
                rng_batch (numpy.Randomstate): Random permutation for batches

            Return:
                training_results (dict): Contains losses of training and simulation losses.
                best_params (list): list of parameter weights.
                best_coeff_mask (numpy.array): If threshold is activated coeff_mask set all values below a threshold
                to zero. Multiplied with the SINDy library it becomes sparse partial differential equations.
            """
        index = 0
        run = True
        while run:
            run = self.check_conditions(index)
            if(run):
                idx = self.get_idx(rng_batch)
                for i in range(self.hyper_params['num_batches']):
                    batch_idx = idx[i * self.hyper_params['batch_size']:(i + 1) * self.hyper_params['batch_size']]
                    self.model.forward(index, self.X[batch_idx], self.dX[batch_idx], self.start_refinement)
                self.evaluate(index)
                self.thresholding(index)
                index += 1
        if self.hyper_params['stop_criterion'] == 'epochs':
            self.best_params = self.model.get_params()
            self.best_coeff_mask = self.model.coeff_mask
        return self.training_results, self.best_params, self.best_coeff_mask

    def check_conditions(self, index):
        """
            Checks if coditions are fulfilled.

            Arguments:
                index (int): current epoch.

            Return:
                run (bool): If conditions are not fulfilled, training stops.
            """
        if self.hyper_params['stop_criterion'] == 'epochs' and index <= self.hyper_params['epochs'] + self.hyper_params['epochs_refinement']:
            run = True
        elif self.hyper_params['stop_criterion'] == 'best_score' and self.number_best < self.hyper_params['stop_best_score']:
            run = True
        else:
            run = False
        return run

    def get_idx(self, rng_batch):
        """
        Shuffles indices of the training set. Will be used for slicing batches.

        Arguments:
            rng_batch (numpy.Randomstate): Random permutation for batches

        Return:
            idx (numpy.array): Shuffled indices.
        """
        if (self.hyper_params['shuffle']):
            idx = rng_batch.permutation(self.X.shape[0])
        else:
            idx = np.arange(self.X.shape[0])
        return idx

    def evaluate(self, index):
        """
        Calculates the loos terms as defined in "Model-Order reduction as an Inverse Problem".
        Additionaly simulations are made and error of the simulations compared to the reconstructions are calculated.

        Arguments:
            index (int): Current epoch.

        """
        params_all = self.model.get_params()
        params_psi = self.model.get_params_psi()
        coeff_mask = self.model.coeff_mask
        if (index % self.hyper_params['model_eval'] == 0):
            results = self.evaluator.evaluate(params_all, params_psi, coeff_mask)
            losses = self.evaluator.calculate_losses(results, index)
            losses['epoch'] = index
            self.training_results = self.training_results.append(losses, ignore_index=True)
            if self.hyper_params['stop_criterion'] == 'best_score' and index > 3000:
                if losses['val_loss'] < self.best_loss:
                    self.best_loss = losses['val_loss']
                    self.number_best = 0
                    self.best_params = params_all
                    self.best_coeff_mask = coeff_mask
                else:
                    self.number_best += 1
                    print('number_best: ' + str(self.number_best))
            self.evaluator.plot_losses(losses, index)

        if (self.start_refinement == False and self.hyper_params['epochs'] == index and self.hyper_params['refinement']):
            print('Start Refinement by epoch: ' + str(index))
            print()
            self.start_refinement = True

    def thresholding(self, index):
        """
        If threholding is activated, entries in sindy coeff that are below "threshold" are set fix to 0.
        coeff_mask memorizes the 0 entries.

        Arguments:
            index (int): Current epoch.

        """
        if self.hyper_params['activate_thresholding'] and index % self.hyper_params['threshold_freq'] == 0 and index > 0:
            if (self.hyper_params['stop_criterion'] == 'epochs' and index < self.hyper_params['epochs']) or self.hyper_params['stop_criterion'] == 'best_score':
                sindy_coeff = self.model.get_params()[-1][0]
                self.model.coeff_mask = (jnp.abs(sindy_coeff) > self.hyper_params['threshold']) * 1.0
                print('Active Terms: ' + str(self.model.coeff_mask.sum()))