import jax
import jax.numpy as jnp
import jax.random as jax_random
from jax import vmap, jit
from jax.nn.initializers import glorot_normal, normal
import pdb
import numpy as np

def serial(two_NN):
    layers1, layers2 = two_NN
    """Combinator for composing layers in serial.
    Args:
      two_NN: takes two list. Every list is a sequence of layers
      where each layer has a (init_fun, apply_fun) pair.
      width_z_pic: int. Width and height of latentspace, only true when using CNN.
    Returns:
      init_fun, apply_decode, apply_f_sim, nlayers_decode
    """
    init_funs1, apply_funs1 = zip(*layers1)
    init_funs2, apply_funs2 = zip(*layers2)

    def init_fun(rng, input_shape):
        z_dim_shape = input_shape
        params = []
        for i, init_fun in enumerate(init_funs1):
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = z_dim_shape
        for init_fun in init_funs2:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_psi(params, inputs):
        for fun, param in zip(apply_funs1, params):
            inputs = fun(param, inputs)
        return inputs

    def apply_g(params, inputs, **kwargs):
        for fun, param in zip(apply_funs2, params):
            inputs = fun(param, inputs)
        return inputs

    return init_fun, apply_psi, apply_g

def serial_autoencoder(two_NN, hyper_params):
    layers1, layers2 = two_NN
    """Combinator for composing layers in serial.

    Args:
      two_NN: takes two list. Every list is a sequence of layers
      where each layer has a (init_fun, apply_fun) pair.
      width_z_pic: int. Width and height of latentspace, only true when using CNN.

    Returns:
      init_fun, apply_encode, apply_decode, apply_f_sim, nlayers_decode
    """
    init_funs1, apply_funs1 = zip(*layers1)
    init_funs2, apply_funs2 = zip(*layers2)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs1:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        for init_fun in init_funs2:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        params.append((jnp.ones((hyper_params['library_dim'], hyper_params['z_latent'])), ))
        return input_shape, params

    def apply_phi(params, inputs):
        for fun, param in zip(apply_funs1, params):
            inputs = fun(param, inputs)
        return inputs

    def apply_psi(params, inputs):
        for fun, param in zip(apply_funs2, params):
            inputs = fun(param, inputs)
        return inputs
    return init_fun, apply_phi, apply_psi

def serial_InvNet(NN, hyper_params):
    init_funs, apply_funs = zip(*NN)
    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        params.append((jnp.ones((hyper_params['library_dim'], hyper_params['z_latent'])), ))
        return input_shape, params
    def apply_psi(params, inputs):
        for fun, param in zip(apply_funs, params):
            inputs = fun(param, inputs)
        return inputs
    return init_fun, apply_psi


#def LayerNorm():
#    def init_fun(rng, input_shape):
#        beta = jnp.zeros(1)
#        gamma = jnp.ones(1)
#        return input_shape, (beta, gamma)
#    def apply_fun(params, inputs):
#        beta, gamma = params
#        mu, var = jnp.mean(inputs), jnp.var(inputs)
#        lnorm = (inputs - mu) / jnp.sqrt(var + 1e-5)
#        return gamma * lnorm + beta
#    return init_fun, apply_fun

def init_sindy_library(hyper_params):
    @jit
    def sindy_library(z):
        library = [jnp.ones((z.shape[0],))]
        for i in range(hyper_params['z_latent']):
            library.append(z[:, i])
        if hyper_params['poly_order'] > 1:
            for i in range(hyper_params['z_latent']):
                for j in range(i, hyper_params['z_latent']):
                    library.append(jnp.multiply(z[:, i], z[:, j]))
        if hyper_params['poly_order'] > 2:
            for i in range(hyper_params['z_latent']):
                for j in range(i, hyper_params['z_latent']):
                    for k in range(j, hyper_params['z_latent']):
                        z1 = jnp.multiply(z[:, i], z[:, j])
                        library.append(jnp.multiply(z1, z[:, k]))
        if hyper_params['include_sine']:
            for i in range(hyper_params['z_latent']):
                library.append(jnp.sin(z[:, i]))
        return jnp.vstack(library).T
    return sindy_library


def reshape_and_to_numpy(arr, n_samples, n_times):
    arr_reshaped = []
    for x in arr:
        x_reshape = np.asarray(x)[:n_samples*n_times].reshape((n_samples, n_times, -1))
        arr_reshaped.append(x_reshape)
    return arr_reshaped

def to_numpy_params_list(params):
    params_np = []
    for param in params:
        if(len(param) == 0):
            params_np.append(())
        else:
            layer = ()
            for j, weight in enumerate(param):
                    layer += (np.asarray(weight, dtype=np.float32), )
            params_np.append(layer)
    return params_np