import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data(cnn=False):
    X = pd.read_csv('data/X.csv', index_col=0).values
    dX = pd.read_csv('data/dX.csv', index_col=0).values
    z_ref = pd.read_csv('data/z_ref.csv', index_col=0).values
    dz_ref = pd.read_csv('data/dz_ref.csv', index_col=0).values
    z_ref_both = jnp.hstack((z_ref, dz_ref))
    if(cnn):
        X = np.hstack((X, np.zeros((X.shape[0], 13))))
        dX = np.hstack((dX, np.zeros((dX.shape[0], 13))))
    return X, dX, z_ref_both

def display_multiple_img(images, i, name, rows=1, cols=1):
    # images: dictonary
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for ind, title in enumerate(images):
        ax.ravel()[ind].plot(images[title])
        ax.ravel()[ind].set_title(title)
        #ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.savefig('pictures/' + str(i) + '_' + str(name) + '.png')
    plt.close()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def loss_dummy_model(X, dX, eta1):
    x_mean = jnp.mean(X, axis=0)
    dx_mean = jnp.mean(dX, axis=0)
    T_recon = jnp.mean(jnp.sum((X - x_mean) ** 2, axis=1))
    T_dx = eta1 * jnp.mean(jnp.sum((X - dx_mean) ** 2, axis=1))
    print('Dummy:  ' + 'T_recon: ' + str(T_recon) + '   ' + 'T_dx: ' + str(T_dx))