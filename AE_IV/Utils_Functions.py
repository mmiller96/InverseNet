import jax.numpy as jnp
import jax.random as jax_random
from jax import vmap
from jax.nn.initializers import glorot_normal, normal

def serial(two_NN):   # if
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

    def apply_psi(params, inputs, signal):
        inputs = jnp.hstack((inputs, signal))
        for fun, param in zip(apply_funs1, params):
            inputs = fun(param, inputs)
        return inputs

    def apply_g(params, inputs, signal, **kwargs):
        inputs = jnp.hstack((inputs, signal))
        for fun, param in zip(apply_funs2, params):
            inputs = fun(param, inputs)
        return inputs

    return init_fun, apply_psi, apply_g

def serial_autoencoder(three_NN):   # if
    layers1, layers2, layers3 = three_NN
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
    init_funs3, apply_funs3 = zip(*layers3)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs1:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = (input_shape[0], input_shape[1] + 4)
        z_dim = input_shape
        for init_fun in init_funs2:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        input_shape = z_dim
        for init_fun in init_funs3:
            rng, layer_rng = jax_random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_phi(params, inputs):
        for fun, param in zip(apply_funs1, params):
            inputs = fun(param, inputs)
        return inputs

    def apply_psi(params, inputs, signal):
        inputs = jnp.hstack((inputs, signal))
        for fun, param in zip(apply_funs1, params):
            inputs = fun(param, inputs)
        return inputs

    def apply_g(params, inputs, signal, **kwargs):
        inputs = jnp.hstack((inputs, signal))
        for fun, param in zip(apply_funs2, params):
            inputs = fun(param, inputs)
        return inputs

    return init_fun, apply_phi, apply_psi, apply_g

def LayerNorm():
    def init_fun(rng, input_shape):
        beta = jnp.zeros(1)
        gamma = jnp.ones(1)
        return input_shape, (beta, gamma)
    def apply_fun(params, inputs):
        beta, gamma = params
        mu, var = jnp.mean(inputs), jnp.var(inputs)
        lnorm = (inputs - mu) / jnp.sqrt(var + 1e-5)
        return gamma * lnorm + beta
    return init_fun, apply_fun

def LayerNormConv():
    def init_fun(rng, input_shape):
        beta = jnp.zeros(input_shape[-1])
        gamma = jnp.ones(input_shape[-1])
        return input_shape, (beta, gamma)

    def apply_fun(params, inputs):
        beta, gamma = params
        mu, var = jnp.mean(inputs, axis=(0, 1, 2)), jnp.var(inputs, axis=(0, 1, 2))
        lnorm = (jnp.reshape(mu, (1, 1, 1, -1)) - inputs)/jnp.sqrt(jnp.reshape(var, (1, 1, 1, -1)) + 1e-5)
        return jnp.reshape(gamma, (1, 1, 1, -1)) * lnorm + jnp.reshape(beta, (1, 1, 1, -1))
    return init_fun, apply_fun

def Reshape(newshape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (newshape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
  return init_fun, apply_fun

def DenseVMAP(out_dim, W_init=glorot_normal(), b_init=normal()):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = jax_random.split(rng)
    W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return jnp.dot(inputs, W) + b
  apply_fun_vmap = vmap(apply_fun, (None, 0))
  return init_fun, apply_fun_vmap


#model_params = [
#            [Dense(25), LayerNorm(), Relu, Reshape((1, 5, 5, 1)),
#             ConvTranspose(16, (6, 6), padding='VALID'), LayerNormConv(), Relu,  # 10x10
#             ConvTranspose(8, (6, 6), padding='VALID'), LayerNormConv(), Relu,  # 15x15
#             ConvTranspose(1, (6, 6), padding='VALID'), LayerNormConv(), Reshape((400,))],  # 20x20
#            [Dense(25), LayerNorm(), Relu, Reshape((1, 5, 5, 1)),
#             Conv(16, (4, 4), padding='same'), LayerNormConv(), Relu,
#             Conv(8, (3, 3), padding='same'), LayerNormConv(), Relu,
#             Conv(1, (3, 3), padding='same'), LayerNormConv(), Reshape((25,)),  # 2 from Conv before
#             Dense(21)]
#        ]