import jax
import numpy.random as npr
import pdb

from Utils import display_multiple_img, get_data
from Utils_Models import get_model
from Train import Trainer



if __name__ == '__main__':
    rng = jax.random.PRNGKey(-1)
    rng_batch = npr.RandomState(0)
    X, dX, z_ref = get_data()
    #display_multiple_img({'P_I: ': X[:, 0], 'dP_I: ': dX[:, 0], 'z_P_I: ': z_ref[:, 0], 'dz_P_I: ': z_ref[:, 2],
    #                      'Q_I: ': X[:, 1], 'dQ_I: ': dX[:, 1], 'z_Q_I: ': z_ref[:, 1], 'dz_Q_I: ': z_ref[:, 3]}, 0,
    #                     'X_real', rows=2, cols=4)
    model_name = 'AE'
    if(model_name == 'AE'):
        hyper_params = {'lr': 0.001,  'epochs': 50, 'batch_size': 128, 'z_latent': 20,               # eta1 = dx
                        'eta1': 10.0, 'eta2': 1e-3, 'eta3': 10e-5, 'x_dim': 159}                     # eta2 = dz
                                                                                                     # eta3 = regularization
    elif(model_name == 'IV'):
        hyper_params = {'lr': 0.001, 'epochs': 110, 'batch_size': 128, 'z_latent': 20,
                        'eta1': 10.0, 'eta2': 1e-4, 'eta3': 1e-5, 'alpha': 0.050, 'steps_inner': 10}
    else:
        raise NameError('Wrong model name')

    model = get_model(model_name, hyper_params, rng)
    step_sample = 50

    trainer = Trainer(model, hyper_params, step_sample, shuffle=False)
    z_pred, x_pred = trainer.fit(X, dX, z_ref, rng_batch)
    pdb.set_trace()




