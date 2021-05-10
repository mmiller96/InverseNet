import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from autoencoder import full_network, define_loss
#from training import create_feed_dictionary
#from sindy_utils import library_size
import os
import tensorflow as tf
#tf.set_random_seed()
import jax
import jax.numpy as jnp

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
        if isinstance(images[title], list):
            for j, arr in enumerate(images[title]):
                #import pdb
                #pdb.set_trace()
                if(j>0): ax.ravel()[ind].plot(arr, linestyle='dotted')
                else: ax.ravel()[ind].plot(arr, linestyle='solid')
        else:
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

def inplace_model_parameters(model, training_data, hyper_params):
    params = get_params_tf(training_data, hyper_params)
    network = full_network(params)
    loss, losses, loss_refinement = define_loss(network, params)
    print(network.keys())
    print(params['epoch_size'] // params['batch_size'])
    batch_idx = np.arange(0 * params['batch_size'], (0 + 1) * params['batch_size'])
    train_dict = create_feed_dictionary(training_data, params, idxs=batch_idx)
    train_dict.pop('learning_rate:0', None)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x, dx, z, dz, x_rec, dx_rec, encoder_weights, encoder_biases, decoder_weights, decoder_biases, Theta, sindy_coefficients, dz_pred, Ls, L = sess.run(
            [network['x'], network['dx'], network['z'], network['dz'], network['x_decode'], network['dx_decode'],
             network['encoder_weights'], network['encoder_biases'], network['decoder_weights'],
             network['decoder_biases'], network['Theta'], network['sindy_coefficients'], network['dz_predict'],
             losses, loss], feed_dict=train_dict)

        tf_params = {'x': x, 'dx': dx, 'z': z, 'dz': dz, 'x_decode': x_rec, 'dx_decode': dx_rec,
                     'encoder_weights': encoder_weights, 'encoder_biases': encoder_biases,
                     'decoder_weights': decoder_weights,
                     'decoder_biases': decoder_biases, 'Theta': Theta, 'sindy_coeff': sindy_coefficients,
                     'dz_pred': dz_pred,
                     'losses': Ls, 'loss': L}
        train_dict['learning_rate:0'] = 0.001
        for i in range(params['max_epochs'] + params['refinement_epochs']):
            if (i % params['print_frequency']== 0):
                batch_idx = np.arange(0 * params['batch_size'], (0 + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idx)
                Ls, L = sess.run([losses, loss], feed_dict=train_dict)
                print(str(i) + ': ' + str(L) + '    ' + str(Ls))
            for j in range(params['epoch_size'] // params['batch_size']):
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op, feed_dict=train_dict)
            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0) and i<5000:
                params['coefficient_mask'] = np.abs(sess.run(network['sindy_coefficients'])) > params['coefficient_threshold']
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
        x, dx, z, dz, x_rec, dx_rec, encoder_weights, encoder_biases, decoder_weights, decoder_biases, Theta, sindy_coefficients, dz_pred, Ls, L = sess.run(
            [network['x'], network['dx'], network['z'], network['dz'], network['x_decode'],
             network['dx_decode'],
             network['encoder_weights'], network['encoder_biases'], network['decoder_weights'],
             network['decoder_biases'], network['Theta'], network['sindy_coefficients'], network['dz_predict'],
             losses, loss], feed_dict=train_dict)

        tf_params2 = {'x': x, 'dx': dx, 'z': z, 'dz': dz, 'x_decode': x_rec, 'dx_decode': dx_rec,
                      'encoder_weights': encoder_weights, 'encoder_biases': encoder_biases,
                      'decoder_weights': decoder_weights,
                      'decoder_biases': decoder_biases, 'Theta': Theta, 'sindy_coeff': sindy_coefficients,
                      'dz_pred': dz_pred,
                      'losses': Ls, 'loss': L}


            #sess.run(train_op, feed_dict=train_dict)
    init_params = model.get_params()
    model = inplace_params(model, init_params, tf_params)
    return model, tf_params, tf_params2, batch_idx

def get_params_tf(training_data, hyper_params):
    params = {}

    params['input_dim'] = 128
    params['latent_dim'] = 3
    params['model_order'] = 1
    params['poly_order'] = 3
    params['include_sine'] = False
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

    # sequential thresholding parameters
    params['sequential_thresholding'] = True
    params['coefficient_threshold'] = hyper_params['threshold']
    params['threshold_frequency'] = hyper_params['threshold_freq']
    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
    params['coefficient_initialization'] = 'constant'

    # loss function weighting
    params['loss_weight_decoder'] = 1.0
    params['loss_weight_sindy_z'] = hyper_params['eta2']
    params['loss_weight_sindy_x'] = hyper_params['eta1']
    params['loss_weight_sindy_regularization'] = hyper_params['eta3']

    params['activation'] = 'sigmoid'
    params['widths'] = [64, 32]

    # training parameters
    params['epoch_size'] = training_data['x'].shape[0]
    params['batch_size'] = hyper_params['batch_size']
    params['learning_rate'] = hyper_params['lr']

    params['data_path'] = os.getcwd() + '/'
    params['print_progress'] = True
    params['print_frequency'] = hyper_params['model_eval']

    # training time cutoffs
    params['max_epochs'] = hyper_params['epochs']
    params['refinement_epochs'] = hyper_params['epochs_refinement']
    return params

def inplace_params(model, init_params, tf_params, model_name):
    encoder_weights, encoder_biases, decoder_weights, decoder_biases = tf_params['encoder_weights'], tf_params['encoder_biases'], tf_params['decoder_weights'], tf_params['decoder_biases']
    #init_params = model.get_params()
    if('AE'):
        init_params[0] = list(init_params[0])
        init_params[2] = list(init_params[2])
        init_params[4] = list(init_params[4])
        init_params[5] = list(init_params[5])
        init_params[7] = list(init_params[7])
        init_params[9] = list(init_params[9])

        init_params[0][0] = jax.ops.index_update(init_params[0][0], jax.ops.index[:], encoder_weights[0])
        init_params[0][1] = jax.ops.index_update(init_params[0][1], jax.ops.index[:], encoder_biases[0])
        init_params[2][0] = jax.ops.index_update(init_params[2][0], jax.ops.index[:], encoder_weights[1])
        init_params[2][1] = jax.ops.index_update(init_params[2][1], jax.ops.index[:], encoder_biases[1])
        init_params[4][0] = jax.ops.index_update(init_params[4][0], jax.ops.index[:], encoder_weights[2])
        init_params[4][1] = jax.ops.index_update(init_params[4][1], jax.ops.index[:], encoder_biases[2])

        init_params[0][0] = jax.ops.index_update(init_params[0][0], jax.ops.index[:], encoder_weights[0])
        init_params[0][1] = jax.ops.index_update(init_params[0][1], jax.ops.index[:], encoder_biases[0])
        init_params[2][0] = jax.ops.index_update(init_params[2][0], jax.ops.index[:], encoder_weights[1])
        init_params[2][1] = jax.ops.index_update(init_params[2][1], jax.ops.index[:], encoder_biases[1])
        init_params[4][0] = jax.ops.index_update(init_params[4][0], jax.ops.index[:], encoder_weights[2])
        init_params[4][1] = jax.ops.index_update(init_params[4][1], jax.ops.index[:], encoder_biases[2])

        # init_params[5] = tuple(init_params[5])
        # init_params[7] = tuple(init_params[7])
        # init_params[9] = tuple(init_params[9])
    else:
        init_params[0] = list(init_params[0])
        init_params[2] = list(init_params[2])
        init_params[4] = list(init_params[4])

        init_params[0][0] = jax.ops.index_update(init_params[0][0], jax.ops.index[:], decoder_weights[0])
        init_params[0][1] = jax.ops.index_update(init_params[0][1], jax.ops.index[:], decoder_biases[0])
        init_params[2][0] = jax.ops.index_update(init_params[2][0], jax.ops.index[:], decoder_weights[1])
        init_params[2][1] = jax.ops.index_update(init_params[2][1], jax.ops.index[:], decoder_biases[1])
        init_params[4][0] = jax.ops.index_update(init_params[4][0], jax.ops.index[:], decoder_weights[2])
        init_params[4][1] = jax.ops.index_update(init_params[4][1], jax.ops.index[:], decoder_biases[2])

        init_params[0] = tuple(init_params[0])
        init_params[2] = tuple(init_params[2])
        init_params[4] = tuple(init_params[4])


    model.opt_state = model.opt_init(init_params)
    return model

def calc_tf_autoencoder(model, training_data, validation_data):
    import sys
    sys.path.append("../../src")
    import os
    import datetime
    import pandas as pd
    # import numpy as np
    from example_lorenz import get_lorenz_data
    from sindy_utils import library_size
    from training import train_network
    import tensorflow as tf

    params = get_params_tf(training_data)

    num_experiments = 1
    df = pd.DataFrame()
    for i in range(num_experiments):
        print('EXPERIMENT %d' % i)

        params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

        params['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        tf.reset_default_graph()

        results_dict, params_init = train_network(training_data, validation_data, params)
        df = df.append({**results_dict, **params}, ignore_index=True)

    df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
    model = inplace_params(model, params_init)
    return model