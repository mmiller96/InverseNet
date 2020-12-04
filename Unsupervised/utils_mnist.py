from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

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
