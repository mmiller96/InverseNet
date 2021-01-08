from jax.experimental.stax import Dense, Relu, ConvTranspose, Conv, Sigmoid
from Utils_Functions import LayerNorm, LayerNormConv, Reshape
from Model_AutoEncoder import AutoEncoder
from Model_InvNet import InvNet

def get_model_params(model_name, hyper_params):
    if(model_name == 'AE'):
        model_params = [
            [Dense(120), LayerNorm(), Relu,
             Dense(64), LayerNorm(), Relu,
             Dense(hyper_params['z_latent'])],
            [Dense(64), LayerNorm(), Relu,
             Dense(120), LayerNorm(), Relu,
             Dense(159), Sigmoid],
            [Dense(64), LayerNorm(), Relu,
             Dense(128), LayerNorm(), Relu,
             Dense(hyper_params['z_latent'])]
        ]
    elif(model_name == 'IV'):
        model_params = [
            [Dense(64), LayerNorm(), Relu,
             Dense(120), LayerNorm(), Relu,
             Dense(159), Sigmoid],
            [Dense(64), LayerNorm(), Relu,
             Dense(128), LayerNorm(), Relu,
             Dense(hyper_params['z_latent'])]
        ]
    else:
        raise NameError('Wrong model name')
    return model_params

def get_model(model_name, hyper_params, rng):
    check_if_correct(model_name, hyper_params)
    model_params = get_model_params(model_name, hyper_params)
    if(model_name == 'AE'):
        hyper_params['n_phi'] = len(model_params[0])
        hyper_params['n_psi'] = len(model_params[1])
        model = AutoEncoder(model_params, hyper_params, rng)
    elif(model_name == 'IV'):
        hyper_params['n_psi'] = len(model_params[0])
        model = InvNet(model_params, hyper_params, rng)
    else:
        raise NameError('Wrong model name')
    return model

def check_if_correct(model_name, hyper_params):
    if(model_name == 'AE'):
        needed_params = ['lr', 'epochs', 'batch_size', 'z_latent', 'eta1', 'eta2', 'eta3', 'x_dim']
    elif(model_name == 'IV'):
        needed_params = ['lr', 'epochs', 'batch_size', 'z_latent', 'eta1', 'eta2', 'eta3', 'alpha', 'steps_inner']
    else:
        raise NameError('Wrong model name')
    for name_param in needed_params:
        if(name_param in hyper_params):
            pass
        else:
            message = str(name_param) + ' is not a defined hyperparameter for' + model_name
            raise Exception(message)