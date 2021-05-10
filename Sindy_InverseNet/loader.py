from jax.experimental.stax import Dense, Relu, ConvTranspose, Conv, Sigmoid
from jax.nn.initializers import zeros
import jax.numpy as jnp
import jax
import numpy as np

from model_autoencoder import Autoencoder
from model_inversenet import Inversenet
from sindy_utils import library_size
from example_lorenz import get_lorenz_data

class Loader():
    def __init__(self, hyper_params):
         self.hyper_params = hyper_params

    def create_lorenz_data(self):
        """
            Create dataset from a lorenz atractor. Simulations are done for 5s, 20ms steps.
            Simulations are stacked to big matrices X, dX, X_eval, dX_eval

            Returns:
                tuple: dataset of stacked simulations (X, dX, X_eval, dX_eval)

         """
        training_data = get_lorenz_data(self.hyper_params['num_train_examples'], noise_strength=1e-6)
        validation_data = get_lorenz_data(self.hyper_params['num_val_examples'], noise_strength=1e-6)

        X, dX = np.array(training_data['x']), np.array(training_data['dx'])
        X_eval, dX_eval = np.array(validation_data['x']), np.array(validation_data['dx'])
        t = np.asarray(training_data['t'])

        self.hyper_params['num_batches'], _ = divmod(X.shape[0], self.hyper_params['batch_size'])
        self.hyper_params['x_dim'] = X.shape[1]
        return X, dX, X_eval, dX_eval, t

    def create_model(self, rng):
        """
            Create a model (Autoencoder / InverseNet) with random weights.

            Args:
                rng (jax.random.PRNGKey): Random key for initializing parameter weights
            Returns:
                model, hyper_params: autoencoder or InverseNet, hyper_params with additional values

         """
        model_params = self.create_model_params()
        if (self.hyper_params['model_name'] == 'AE'):
            self.hyper_params['n_phi'] = len(model_params[0])
            self.hyper_params['n_psi'] = len(model_params[1])
            model = Autoencoder(model_params, self.hyper_params, rng)
        elif (self.hyper_params['model_name'] == 'IV'):
            self.hyper_params['n_psi'] = len(model_params)
            model = Inversenet(model_params, self.hyper_params, rng)
        else:
            raise NameError('Wrong model name')
        return model, self.hyper_params

    def create_model_params(self):
        """
            Random Weights for Autoencoder / InverseNet. These parameters are trained in the models.

            Returns:
                model_params (list): Contains numpy.arrays. These are different layers and activation functions.

         """
        if (self.hyper_params['model_name'] == 'AE'):
            model_params = [
                [Dense(64, b_init=zeros), Sigmoid,
                 Dense(32, b_init=zeros), Sigmoid,
                 Dense(self.hyper_params['z_latent'], b_init=zeros)],
                [Dense(32, b_init=zeros), Sigmoid,
                 Dense(64, b_init=zeros), Sigmoid,
                 Dense(self.hyper_params['x_dim'], b_init=zeros)]
            ]
        elif (self.hyper_params['model_name'] == 'IV'):
            model_params = [
                Dense(32, b_init=zeros), Sigmoid,
                Dense(64, b_init=zeros), Sigmoid,
                Dense(self.hyper_params['x_dim'], b_init=zeros)]
        else:
            raise NameError('Wrong model name')
        return model_params