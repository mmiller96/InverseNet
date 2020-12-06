from model_InvNet import invNet_jax
from utils_mnist import load_dataset_scipy, display_multiple_img, serial

import matplotlib.pyplot as plt
import pdb
import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, Relu, Sigmoid, Softmax
from sklearn.metrics import accuracy_score

def evaluate(X, y, y_oh, model_jax, samples, j, show_train=False):
    z = jnp.zeros((X.shape[0], model_jax.z_latent))
    params = model_jax.get_params(model_jax.opt_state)
    params_sim = params[model_jax.nlayers_decode:]
    T = model_jax.T_vec(params, X, y_oh, z)
    T_class = model_jax.T_classify_vec(params, X, y_oh, z)
    z_opt = model_jax.encode_vec(params, X, z)
    y_pred = model_jax.f_sim_vec(params_sim, z_opt)
    X_recon = model_jax.decode_vec(params, z_opt)
    X, X_recon = X.reshape(len(X), 8, 8), X_recon.reshape((len(X_recon), 8, 8))
    if(show_train):
        X = X[::samples]
        X = X[:9]
        images = {'Class' + str(i): X[i, :, :] for i in range(9)}
        name = 'train'
    else:
        X_recon = X_recon[::samples]
        X_recon = X_recon[:9]
        images = {'Class' + str(i): X_recon[i, :, :] for i in range(9)}
        name = 'recon'
    display_multiple_img(images, j, name, 3, 3)
    score = accuracy_score(y, jnp.argmax(y_pred, axis=1))
    print(str(i) + ' T: ' + str(T.sum()) + '   ' + 'T_class: ' + str(T_class.sum()) + '   ACC: ' + str(score))


if __name__ == '__main__':
    rng = jax.random.PRNGKey(10)
    alpha = 0.1
    steps_inner = 10
    z_latent = 16
    lr = 0.002
    iterations = 2500
    etha = 0.1
    train_samples = 3
    val_samples = 50

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh = load_dataset_scipy(train_samples=train_samples, val_test_samples=val_samples)

    net_init, net_decode, net_f_sim, nlayers_decode = serial([
        [Dense(16), Relu,
        Dense(32), Relu,
        Dense(64), Sigmoid],
        [Dense(10), Softmax]
    ])
    _, init_params = net_init(rng, (-1, z_latent))
    model_jax = invNet_jax(init_params, net_decode, net_f_sim, nlayers_decode, alpha, etha, steps_inner, z_latent, lr)

    for i in range(iterations):
        model_jax.forward(i, X_train, y_train_oh)
        if(i%50 == 0):
            if(i==0):
                evaluate(X_val, y_val, y_val_oh, model_jax, val_samples, i, show_train=True)
            evaluate(X_val, y_val, y_val_oh, model_jax, val_samples, i)
