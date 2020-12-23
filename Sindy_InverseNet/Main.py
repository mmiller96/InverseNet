import jax
import numpy.random as npr
from jax.experimental.stax import Dense, Relu, ConvTranspose, Conv

from Model_InvNet import InvNet
from Utils import display_multiple_img, get_data, loss_dummy_model
from Utils_Functions import LayerNorm, LayerNormConv, Reshape

def get_model_params(cnn=False):
    if (cnn):
        model_params = [
            [Dense(25), LayerNorm(), Relu, Reshape((1, 5, 5, 1)),
             ConvTranspose(16, (6, 6), padding='VALID'), LayerNormConv(), Relu,  # 10x10
             ConvTranspose(8, (6, 6), padding='VALID'), LayerNormConv(), Relu,  # 15x15
             ConvTranspose(1, (6, 6), padding='VALID'), LayerNormConv(), Reshape((400,))],  # 20x20
            [Dense(25), LayerNorm(), Relu, Reshape((1, 5, 5, 1)),
             Conv(16, (4, 4), padding='same'), LayerNormConv(), Relu,
             Conv(8, (3, 3), padding='same'), LayerNormConv(), Relu,
             Conv(1, (3, 3), padding='same'), LayerNormConv(), Reshape((25,)),  # 2 from Conv before
             Dense(21)]
        ]
    else:
        model_params = [
            [Dense(64), LayerNorm(), Relu,
             Dense(128), LayerNorm(), Relu,
             Dense(387)],
            [Dense(64), LayerNorm(), Relu,
             Dense(128), LayerNorm(), Relu,
             Dense(64), LayerNorm(), Relu,
             Dense(hyper_params['z_latent'])]
        ]
    return model_params


if __name__ == '__main__':
    rng = jax.random.PRNGKey(-1)
    rng_batch = npr.RandomState(0)

    cnn = False
    X, dX, z_ref = get_data(cnn=cnn)
    display_multiple_img({'P_I: ': X[:, 0], 'dP_I: ': dX[:, 0], 'z_P_I: ': z_ref[:, 0], 'dz_P_I: ': z_ref[:, 2],
                          'Q_I: ': X[:, 1], 'dQ_I: ': dX[:, 1], 'z_Q_I: ': z_ref[:, 1], 'dz_Q_I: ': z_ref[:, 3]}, 0,
                         'X_real', rows=2, cols=4)
    hyper_params = {'lr': 0.001, 'alpha': 0.002, 'steps_inner': 10, 'epochs': 20, 'batch_size': 128,
                    'z_latent': 21, 'eta1': 10e-2, 'eta2': 10e-2, 'eta3': 10e-3}
    model_params = get_model_params(cnn=cnn)
    hyper_params['n_psi'] = len(model_params[0])
    model = InvNet(model_params, hyper_params, rng)

    step_sample = 50
    loss_dummy_model(X[::step_sample], dX[::step_sample], hyper_params['eta1'])
    num_batches, _ = divmod(X.shape[0], hyper_params['batch_size'])
    for j in range(hyper_params['epochs']):
        idx = rng_batch.permutation(X.shape[0])
        for i in range(num_batches):
            batch_idx = idx[i * hyper_params['batch_size']:(i + 1) * hyper_params['batch_size']]
            model.forward(i, X[batch_idx], dX[batch_idx], z_ref[batch_idx])
        model.evaluate(X[::step_sample], dX[::step_sample], z_ref[::step_sample], j)
        if (j % 5 == 0): model.save(j)
    z_pred, x_pred = model.predict_45s(X, z_ref)
