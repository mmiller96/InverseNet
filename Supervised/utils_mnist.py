from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from jax import random

def serial(two_NN):
    layers1, layers2 = two_NN
    """Combinator for composing layers in serial.

    Args:
      two_NN: takes two list. Every list is a sequence of layers
      where each layer has a (init_fun, apply_fun) pair.

    Returns:
      init_fun, apply_decode, apply_f_sim, nlayers_decode
    """
    nlayers1 = len(layers1)
    init_funs1, apply_funs1 = zip(*layers1)

    nlayers2 = len(layers2)
    init_funs2, apply_funs2 = zip(*layers2)

    def init_fun(rng, input_shape):
        z_dim_shape = input_shape
        params = []
        for init_fun in init_funs1:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = z_dim_shape
        for init_fun in init_funs2:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_decode(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers1) if rng is not None else (None,) * nlayers1
        for fun, param, rng in zip(apply_funs1, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    def apply_f_sim(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers2) if rng is not None else (None,) * nlayers2
        for fun, param, rng in zip(apply_funs2, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    return init_fun, apply_decode, apply_f_sim, nlayers1

def cast_oh(y):
    y_oh = np.zeros((y.size, y.max() + 1))
    y_oh[np.arange(y.size), y] = 1
    return y_oh

def load_dataset_scipy(train_samples=3, val_test_samples=3):
    # load train, val and test set
    # train_samples: Number of samples per class
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    X_train, X_val, X_test = [], [], []
    for i in range(10):
        X_class = X_trans[y == i, :]
        X_train_class = X_class[:train_samples, :]
        X_val_class = X_class[train_samples:train_samples+val_test_samples, :]
        X_test_class = X_class[train_samples + val_test_samples:train_samples + 2*val_test_samples, :]
        X_train.append(X_train_class)
        X_val.append(X_val_class)
        X_test.append(X_test_class)
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.hstack(X_test)
    y_train = np.repeat(np.arange(10), train_samples)
    y_val = np.repeat(np.arange(10), val_test_samples)
    y_test = np.repeat(np.arange(10), val_test_samples)
    y_train_oh = cast_oh(y_train)
    y_val_oh = cast_oh(y_val)
    y_test_oh = cast_oh(y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh

def display_multiple_img(images, i, name, rows=1, cols=1):
    # images: dictonary
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for ind, title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.savefig('Pictures/' + str(i) + '_' + str(name) + '.png')
    plt.close()
