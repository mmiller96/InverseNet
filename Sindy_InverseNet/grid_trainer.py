import pandas as pd
import os.path
import pdb
import jax
import numpy.random as npr
import pickle
import numpy as np
import h5py

from sindy_utils import library_size
from example_lorenz import get_lorenz_data
from loader import Loader
from train import Trainer

class Grid_Trainer():
    def __init__(self, analysis_folder):
        self.hyper_params_df = self.load_hyper_params(analysis_folder)
        self.data = self.load_data(analysis_folder)
        self.analysis_folder = analysis_folder                              # path folder

    def process(self):
        """
        Trains different model based on a csv-data containing the hyper_params.
        If a model is trained, results, parameters of model, coefficient mask and the hyperparaemters are saved
        in analysis_folder. Models are numbered in the same ordered as they are in the hyper_params.csv.
        """
        rng = jax.random.PRNGKey(-1)
        rng_batch = npr.RandomState(5)
        for j, hyper_params in self.hyper_params_df.iterrows():
            hyper_params = hyper_params.to_dict()
            path_save = self.analysis_folder + '/' + str(j)
            if os.path.isfile(path_save):
                continue
            else:
                loader = Loader(hyper_params)
                train_set = hyper_params['training_set']
                X, dX, X_eval, dX_eval, t = self.data[train_set]
                loader.hyper_params['num_batches'], _ = divmod(X.shape[0], loader.hyper_params['batch_size'])
                loader.hyper_params['x_dim'] = X.shape[1]
                model, hyper_params = loader.create_model(rng)
                trainer = Trainer(X, dX, X_eval, dX_eval, t, model, hyper_params)
                results, params, coeff_mask = trainer.fit(rng_batch)
                with open(path_save, 'wb') as fp:
                    pickle.dump([results, params, coeff_mask, hyper_params], fp)


    def load_hyper_params(self, analysis_folder):
        """ Load hyper_params or if not exist create hyper_params.

            Args:
            analysis_folder (str): folder where the hyper_params are saved

        Returns:
            pandas.Dataframe: hyper_params for different models.

         """
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)
        path_df = analysis_folder + '/hyper_params.csv'
        if os.path.isfile(path_df):
            print('load file...')
            data_frame = pd.read_csv(path_df, index_col=0)
        else:
            print('create file...')
            data_frame = self.create_hyper_parameters(analysis_folder)
            data_frame.to_csv(path_df)
        return data_frame


    def create_hyper_parameters(self, analysis_folder):
        """ Here the hyper_params are defined. For a new data frame with different hyper_params,
            create an elif statemnet with the corresponding hyper_params.


        Args:
            analysis_folder (str): folder where the hyper_params are saved

        Returns:
            pandas.Dataframe: hyper_params for different models.

        """

        if analysis_folder == 'analysis_alpha':
            alpha = np.repeat(np.array([0.1, 0.01, 0.001]), 3)
            hyper_params_grid = {'alpha': alpha}
            
        elif analysis_folder == 'analysis_alpha2':
            hyper_params_grid = {'alpha': [0.1, 0.01, 0.001], 'stop_criterion': ['epochs'], 'eta3': [0.0], 'activate_thresholding': [False],
                                 'model_eval': [500], 'num_val_examples': [100], 'epochs': [60000]}

        elif analysis_folder == 'analysis_depth':
            hyper_params_grid = {'steps_inner': [0, 5, 10, 20], 'model_name': ['AE', 'IV', 'IV', 'IV'],
                                 'stop_criterion': ['epochs'], 'eta3': [0.0], 'activate_thresholding': [False],
                                 'model_eval': [500], 'num_val_examples': [100], 'epochs': [100000]}

        elif analysis_folder == 'analysis_samples':
            training_set = np.repeat(np.array([0, 1, 2, 3, 4]), 10)
            batch_size1 = np.repeat(np.array([250, 750]), 10)
            batch_size2 = np.repeat(1000, 30)
            batch_size = np.hstack((batch_size1, batch_size2))
            num_train_examples = np.repeat(np.array([1, 3, 10, 30, 100]), 10)
            train_num_simulations1 = np.repeat(np.array([1, 3, 10]), 10)
            train_num_simulations2 = np.repeat(np.array([20]), 20)
            train_num_simulations = np.hstack((train_num_simulations1, train_num_simulations2))
            model_name = np.repeat(np.tile(['AE', 'IV'], 5), 5)
            hyper_params_grid = {'steps_inner': [20], 'model_name': model_name, 'batch_size': batch_size,
                                 'num_train_examples': num_train_examples, 'training_set': training_set,
                                 'train_num_simulations': train_num_simulations}
        else:
            raise Exception('No such directory!')
        df = self.create_data_frame(hyper_params_grid)
        return df

        '''
        
        
        
        
        df = pd.DataFrame(
            columns=['stop_criterion', 'lr', 'epochs', 'epochs_refinement', 'batch_size', 'z_latent', 'num_train_examples', 'num_val_examples', 'eta1', 'eta2', 'eta3', 'poly_order', 'model_eval',
                     'library_dim', 'step_sample', 'include_sine', 'shuffle', 'threshold', 'threshold_freq', 'num_simulations', 'activate_thresholding', 'model_name', 'alpha', 'steps_inner', 'stop_best_score'
                     'refinement', 'train_num_simulations'])
        if analysis_name == 'analysis10':
            model_names_list = ['IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV',
                           'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV',
                           'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV', 'IV',
                           'AE', 'AE', 'AE']
            alpha_list = [0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03,
                          0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03,
                          0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03,
                          None, None, None]
            steps_inner_list = [5, 5, 5, 5, 5, 5, 5, 5, 5,
                                10, 10, 10, 10, 10, 10, 10, 10, 10,
                                20, 20, 20, 20, 20, 20, 20, 20, 20,
                                None, None, None]
            hyper_params_fix = {'stop_criterion': 'best_score', 'lr': 1e-3,  # 'epochs': 0, 'epochs_refinement': 0,
                                'batch_size': 1000, 'z_latent': 3, 'num_val_examples': 50,
                                'eta1': 1e-4, 'eta2': 1e-5, 'eta3': 0.0, 'poly_order': 3, 'model_eval': 200,
                                'step_sample': 10, 'include_sine': False, 'shuffle': True,
                                'threshold': 0.05, 'threshold_freq': 500, 'num_simulations': 20,
                                'activate_thresholding': False, 'refinement': False, 'stop_best_score': 15, 'num_train_examples': 100}

            hyper_params_fix['library_dim'] = library_size(hyper_params_fix['z_latent'], hyper_params_fix['poly_order'],
                                                           hyper_params_fix[
                                                               'include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            for model_name, steps_inner, alpha in zip(model_names_list, steps_inner_list, alpha_list):
                hyper_params = {'alpha': alpha, 'steps_inner': steps_inner, 'model_name': model_name}
                hyper_params['train_num_simulations'] = hyper_params_fix['num_simulations']
                hyper_params.update(hyper_params_fix)
                df = df.append(hyper_params, ignore_index=True)

        elif analysis_name == 'analysis11':
            model_names_list = ['IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE',
                                'IV', 'IV', 'IV', 'IV', 'IV', 'AE', 'AE', 'AE', 'AE', 'AE']

            num_train_examples_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                       5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                       10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                       20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                                       30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
                                       50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                       100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            batch_size_list = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                               750, 750, 750, 750, 750, 750, 750, 750, 750, 750,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                               1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
            hyper_params_fix = {'stop_criterion': 'best_score', 'lr': 1e-3,  # 'epochs': 0, 'epochs_refinement': 0,
                                'z_latent': 3, 'num_val_examples': 50, 'steps_inner': 20, 'alpha': 0.01,
                                'eta1': 1e-4, 'eta2': 1e-5, 'eta3': 1e-6, 'poly_order': 3, 'model_eval': 200,
                                'step_sample': 10, 'include_sine': False, 'shuffle': False,
                                'threshold': 0.05, 'threshold_freq': 500, 'num_simulations': 20,
                                'activate_thresholding': True, 'refinement': False, 'stop_best_score': 15}

            hyper_params_fix['library_dim'] = library_size(hyper_params_fix['z_latent'], hyper_params_fix['poly_order'],
                                                           hyper_params_fix[
                                                               'include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            for model_name, num_train_examples, batch_size in zip(model_names_list, num_train_examples_list, batch_size_list):
                hyper_params = {'num_train_examples': num_train_examples, 'model_name': model_name, 'batch_size': batch_size}
                if(num_train_examples < hyper_params_fix['num_simulations']):
                    hyper_params['train_num_simulations'] = num_train_examples
                else:
                    hyper_params['train_num_simulations'] = hyper_params_fix['num_simulations']
                hyper_params.update(hyper_params_fix)
                df = df.append(hyper_params, ignore_index=True)
        elif analysis_name == 'analysis_test':
            hyper_params = {'stop_criterion': 'best_score', 'model_name': 'IV', 'lr': 1e-3,  'num_train_examples': 10, 'batch_size':500, 'train_num_simulations':10,
                                'z_latent': 3, 'num_val_examples': 50, 'steps_inner': 20, 'alpha': 0.01,
                                'eta1': 1e-4, 'eta2': 1e-5, 'eta3': 1e-5, 'poly_order': 3, 'model_eval': 500,
                                'step_sample': 10, 'include_sine': False, 'shuffle': True,
                                'threshold': 0.05, 'threshold_freq': 500, 'num_simulations': 20,
                                'activate_thresholding': True, 'refinement': False, 'stop_best_score': 5}
            hyper_params['library_dim'] = library_size(hyper_params['z_latent'], hyper_params['poly_order'], hyper_params['include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            df = df.append(hyper_params, ignore_index=True)
        elif analysis_name == 'analysis2':
            steps_inner_list = [5, 5, 5, 20, 20, 20]
            hyper_params_fix = {'stop_criterion': 'epochs', 'lr': 1e-3, 'epochs': 30000, 'epochs_refinement': 0,
                                'batch_size': 1000, 'z_latent': 3, 'num_train_examples': 100, 'num_val_examples': 50,
                                'eta1': 1e-4, 'eta2': 1e-5, 'eta3': 0.0, 'poly_order': 3, 'model_eval': 100,
                                'step_sample': 10, 'include_sine': False, 'shuffle': False,
                                'threshold': 0.1, 'threshold_freq': 200, 'num_simulations': 20,
                                'activate_thresholding': False, 'refinement': False, 'threshold_best_score': 5, 'model_name': 'IV', 'alpha': 0.01}
            hyper_params_fix['library_dim'] = library_size(hyper_params_fix['z_latent'], hyper_params_fix['poly_order'], hyper_params_fix['include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            for steps_inner in steps_inner_list:
                hyper_params = {'steps_inner': steps_inner}
                hyper_params.update(hyper_params_fix)
                df = df.append(hyper_params, ignore_index=True)
        elif analysis_name == 'analysis5':
            model_name_list = ['IV', 'IV', 'IV', 'AE', 'AE']
            num_train_examples_list = [5, 5, 5, 5, 5]
            alpha_list = [0.005, 0.002, 0.002, None, None]
            steps_inner_list = [20, 50, 50, None, None]
            eta3_list = [0.0, 1e-6, 0.0, 1e-6, 0.0]
            hyper_params_fix = {'stop_criterion': 'best_score', 'lr': 1e-3, #'epochs': 0, 'epochs_refinement': 0,
                                'batch_size': 1000, 'z_latent': 3,  'num_val_examples': 50,
                                'eta1': 1e-4, 'eta2': 1e-5, 'poly_order': 3, 'model_eval': 200,
                                'step_sample': 10, 'include_sine': False, 'shuffle': True,
                                'threshold': 0.05, 'threshold_freq': 500, 'num_simulations': 20,
                                'activate_thresholding': False, 'refinement': False, 'stop_best_score': 15}

            hyper_params_fix['library_dim'] = library_size(hyper_params_fix['z_latent'], hyper_params_fix['poly_order'], hyper_params_fix['include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            for model_name, num_train_examples, alpha, steps_inner, eta3 in zip(model_name_list, num_train_examples_list, alpha_list, steps_inner_list, eta3_list):
                hyper_params = {'alpha': alpha, 'steps_inner': steps_inner, 'model_name': model_name, 'num_train_examples': num_train_examples, 'eta3': eta3}
                if hyper_params_fix['num_simulations'] > num_train_examples:
                    hyper_params['train_num_simulations'] = num_train_examples
                else:
                    hyper_params['train_num_simulations'] = hyper_params_fix['num_simulations']
                hyper_params.update(hyper_params_fix)
                df = df.append(hyper_params, ignore_index=True)

        elif analysis_name == 'analysis4':
            alpha_list = [0.005, 0.02, 0.02, 0.05, 0.002, 0.005]
            steps_inner_list = [20, 5, 5, 5, 20, 20]
            hyper_params_fix = {'stop_criterion': 'epochs', 'lr': 1e-3, 'epochs': 30000, 'epochs_refinement': 0,
                                'batch_size': 1000, 'z_latent': 3,  'num_val_examples': 50,
                                'eta1': 1e-4, 'eta2': 1e-5, 'eta3': 0.0, 'poly_order': 3, 'model_eval': 100,
                                'step_sample': 10, 'include_sine': False, 'shuffle': False,
                                'threshold': 0.1, 'threshold_freq': 500, 'num_simulations': 20, 'train_num_simulations':20,
                                'activate_thresholding': False, 'refinement': False, 'stop_best_score': 5, 'model_name': 'IV', 'num_train_examples': 100, }
            hyper_params_fix['library_dim'] = library_size(hyper_params_fix['z_latent'], hyper_params_fix['poly_order'], hyper_params_fix['include_sine'])  # stop_criterion: 'epochs' or 'best_score'
            for alpha, steps_inner in zip(alpha_list, steps_inner_list):
                hyper_params = {'alpha': alpha, 'steps_inner': steps_inner}
                #if hyper_params_fix['num_simulations'] > num_train_examples:
                #    hyper_params['train_num_simulations'] = num_train_examples
                #else:

                hyper_params.update(hyper_params_fix)
                df = df.append(hyper_params, ignore_index=True)
        return df
        '''

    def load_data(self, analysis_folder):
        """ Load hdf5 files or if not exist create hdf5 files. Every file has a dataset (X, dX, X_eval, dX_eval)

                    Args:
                        analysis_folder(str): direction to train the models

                    Returns:
                        list: Contains lists. Every list represent a dataset (X, dX, X_eval, dX_eval)

                    """
        data = []
        first_set = analysis_folder + '/dataset0.hdf5'
        # checks if the first hdf5 exists.
        # If it exists it assumes that all hdf5 files exist and the datasets are loaded.
        # Else the datasets are created.
        if os.path.isfile(first_set):
            for training_set in np.unique(self.hyper_params_df['training_set']):
                analysis_file = analysis_folder + '/dataset' + str(training_set) + '.hdf5'
                with h5py.File(analysis_file, 'r') as f:
                    X = np.array(f['X'])
                    dX = np.array(f['dX'])
                    X_eval = np.array(f['X_eval'])
                    dX_eval = np.array(f['dX_eval'])
                    t = np.array(f['t'])
                data.append([X, dX, X_eval, dX_eval, t])
        else:
            for training_set, idx in np.vstack(np.unique(self.hyper_params_df['training_set'], return_index=True)).T:
                analysis_file = analysis_folder + '/dataset' + str(training_set) + '.hdf5'
                with h5py.File(analysis_file, 'w') as f:
                    loader = Loader(self.hyper_params_df.iloc[idx])
                    X, dX, X_eval, dX_eval, t = loader.create_lorenz_data()
                    dset_X = f.create_dataset('X', data=X)
                    dset_dX = f.create_dataset('dX', data=dX)
                    dset_X_eval = f.create_dataset('X_eval', data=X_eval)
                    dset_dX_eval = f.create_dataset('dX_eval', data=dX_eval)
                    dset_t = f.create_dataset('t', data=t)
                data.append([X, dX, X_eval, dX_eval, t])
        return data

    def create_data_frame(self, hyper_params_grid):
        """Takes a dictionary of hyperparameters. Every entry an be a list or a scaler.
            Every list must have the same length. hyper_params_fix are the default values.
            They can be changed with hyper_params_grid. If they are not changed, for all hyper_params the
            default values are set.

            Args:
                hyper_params_grid (dict): hyperparameters to use.

            Returns:
                pandas.Dataframe: Hyperparameters for different models.

            """

        hyper_params_fix = {'stop_criterion': ['best_score'], 'lr': [1e-3], 'alpha': [0.01], 'epochs': [0],
                            'epochs_refinement': [0], 'model_name': ['IV'], 'steps_inner': [20],
                            'batch_size': [1000], 'z_latent': [3], 'num_train_examples': [50], 'num_val_examples': [50],
                            'eta1': [1e-4], 'eta2': [0], 'eta3': [1e-5], 'poly_order': [3], 'model_eval': [500],
                            'step_sample': [10], 'include_sine': [False], 'shuffle': [True],
                            'threshold': [0.1], 'threshold_freq': [500], 'num_simulations': [20],
                            'activate_thresholding': [True], 'refinement': [False], 'stop_best_score': [6],
                            'training_set': [0], 'train_num_simulations': [20]}
        hyper_params_fix['library_dim'] = [library_size(hyper_params_fix['z_latent'][0],
                                            hyper_params_fix['poly_order'][0], hyper_params_fix['include_sine'][0])]
        n = 0
        # search number of models
        for key in hyper_params_grid.keys():
            hyper_params_fix[key] = hyper_params_grid[key]
            if len(hyper_params_fix[key]) > n:
                n = len(hyper_params_fix[key])
        # set default hyper_params for every model
        for key in hyper_params_fix.keys():
            if len(hyper_params_fix[key]) == n:
                continue
            else:
                hyper_params_fix[key] = np.repeat(hyper_params_fix[key], n)
        df = pd.DataFrame(hyper_params_fix)
        return df