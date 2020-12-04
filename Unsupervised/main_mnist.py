from model_InvNet import invNet_jax
from utils_mnist import load_dataset_scipy, display_multiple_img

import torchvision
import torch
from torch import nn
from torch.optim import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
from torch.autograd import grad
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, Sigmoid
from jax.experimental.optimizers import sgd

def evaluate(X, model_jax, samples, j, show_train=False):
    z = jnp.zeros((X.shape[0], model_jax.z_latent))
    params = model_jax.get_params(model_jax.opt_state)
    T = model_jax.T_vec(params, X, z)
    z_opt = model_jax.encode_vec(params, X, z)
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
    print(str(i) + ': ' + str(T.sum()))


if __name__ == '__main__':
    rng = jax.random.PRNGKey(10)
    alpha = 0.1
    steps_inner = 10
    z_latent = 10
    lr = 0.002
    iterations = 1000

    X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh = load_dataset_scipy()

    net_init, net_apply = stax.serial(
        Dense(16), Relu,
        Dense(32), Relu,
        Dense(64), Sigmoid
    )
    _, init_params = net_init(rng, (-1, 10))
    model_jax = invNet_jax(net_apply, init_params, alpha, steps_inner, z_latent, lr)

    for i in range(iterations):
        model_jax.forward(i, X_train)
        if(i%50 == 0):
            if(i==0):
                evaluate(X_val, model_jax, 3, i, show_train=True)
            evaluate(X_val, model_jax, 3, i)
