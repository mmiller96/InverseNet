from Utils import loss_dummy_model
import pdb
import numpy as np

class Trainer():
    def __init__(self, model, hyper_params, step_sample, save_model_every=5, shuffle=True):
        self.model = model
        self.hyper_params = hyper_params
        self.step_sample = step_sample
        self.save_model_every = save_model_every
        self.shuffle = shuffle

    def fit(self, X, dX, z_ref, rng_batch):
        loss_dummy_model(X[::self.step_sample], dX[::self.step_sample], self.hyper_params['eta1'])
        num_batches, _ = divmod(X.shape[0], self.hyper_params['batch_size'])
        for j in range(self.hyper_params['epochs']):
            if(self.shuffle): idx = rng_batch.permutation(X.shape[0])
            else: idx = np.arange(X.shape[0])
            for i in range(num_batches):
                batch_idx = idx[i * self.hyper_params['batch_size']:(i + 1) * self.hyper_params['batch_size']]
                self.model.forward(i, X[batch_idx], dX[batch_idx], z_ref[batch_idx])
            if(j%5 == 0): self.model.evaluate(X, dX, z_ref, self.step_sample, j)
            if (j % self.save_model_every == 0): self.model.save(j)
        z_pred, x_pred = self.model.predict_45s(X, z_ref)
        return z_pred, x_pred