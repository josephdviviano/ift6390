#!/usr/bin/env python
"""
implement a neural network where you compute the gradients using the formulas
derived in the previous part (including elastic net regularization). You must
not use an existing neural network library, and you must use the derivation of
part 2 (with corresponding variable names, etc). Note that you can reuse the
general learning algorithm structure that we used in the demos, as well as the
functions used to plot the decision functions.
"""
from copy import copy
from sklearn.preprocessing import OneHotEncoder
import argparse
import gzip
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time

# adds a simple logger
logging.basicConfig(level=logging.INFO, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger(os.path.basename(__file__))

class Normalizer:
    def __init__(self):
        self.mean = 0
        self.std = 0

    def normalize(self, X, train=True):
        """normalizes an sample x to have 0 mean and unit standard deviation"""
        if train:
            self.mean = np.mean(X)
            self.std = np.mean(X)

        return((X - self.mean)/self.std)


class Scaler:
    def __init__(self):
        self.minimum = 0
        self.maximum = 1

    def scale(self, X, train=True):
        """scales sample x to be in range [0 1]"""
        if train:
            self.minimum = np.min(X)
            self.maximum = np.max(X)


        return((X - self.minimum)/(self.maximum - self.minimum)*2 - 1)


def load_mnist_raw(path, kind='train'):
    """Load Fashion MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
            offset=16).reshape(len(labels), 784)

    return(images, labels)


def make_mnist_proc(output):

    if os.path.isfile(output):
        LOGGER.debug('preprocessed MNIST already exists, skipping preprocessing')
        return(None)

    split = 50000
    X_train, y_train = load_mnist_raw('data/', kind='train')
    X_test, y_test = load_mnist_raw('data/', kind='t10k')

    X_valid = X_train[split:, :]
    X_train = X_train[:split, :]
    y_valid = y_train[split:]
    y_train = y_train[:split]

    norm_buddy = Normalizer()
    X_train = norm_buddy.normalize(X_train)
    X_valid = norm_buddy.normalize(X_valid, train=False)
    X_test  = norm_buddy.normalize(X_test, train=False)

    data = {"X": {"train": X_train, "valid": X_valid, "test": X_test},
            "y": {"train": y_train, "valid": y_valid, "test": y_test}}

    LOGGER.debug('saving preprocessed MNIST data at {}'.format(output))
    with open(output, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return(data)


def get_circles_data():
    data = np.loadtxt(open('data/circles.txt','r'))
    X = data[:, :2]
    y = data[:, 2]

    X_train = X[:800, :]
    X_valid = X[800:950, :]
    X_test  = X[950:, :]
    y_train = y[:800]
    y_valid = y[800:950]
    y_test  = y[950:]

    data = {"X": {"train": X_train, "valid": X_valid, "test": X_test},
            "y": {"train": y_train, "valid": y_valid, "test": y_test}}

    return(data)


class MLP:
    def __init__(self, **kwargs):

        hyperparameters = {
            'n_i'    : 784,
            'n_h'    : 10,
            'n_o'    : 10,
            'epochs' : 10,
            'k'      : 50,
            'lr'     : 0.01,
            'l1'     : 0,
            'l2'     : 0,
            'stupid_loop': False
        }

        hyperparameters.update(kwargs)

        LOGGER.info('MLP initialized with parameters:')
        for hp in hyperparameters.items():
            LOGGER.info('    {} = {}'.format(hp[0], hp[1]))

        self.n_i = hyperparameters['n_i']       # input dimensions
        self.n_h = hyperparameters['n_h']       # number of hidden units
        self.n_o = hyperparameters['n_o']       # number of output units
        self.epochs = hyperparameters['epochs'] # n epochs
        self.k   = hyperparameters['k']         # minibatch size
        self.lr  = hyperparameters['lr']        # learning rate
        self.l1 = hyperparameters['l1']         # lambda l1
        self.l2 = hyperparameters['l2']         # lambda l2
        self.stupid_loop = hyperparameters['stupid_loop'] # do something stupid

        # negative values of k run in BATCH mode
        if self.k <= 0:
            self.minibatch = False
            LOGGER.info('k={}, using BATCH mode'.format(self.k))
        else:
            self.minibatch = True
            LOGGER.debug('using batch size k={}'.format(self.k))

        # construct network and initialize weights
        self.W1 = self._init_W(self.n_i, self.n_h)
        self.W2 = self._init_W(self.n_h, self.n_o)
        self.b1 = np.zeros(self.n_h)
        self.b2 = np.zeros(self.n_o)

        # for softmax loss
        self.onehot = OneHotEncoder(sparse=False, n_values=self.n_o)

        # gradients
        self.dL_W1 = 0
        self.dL_b1 = 0
        self.dL_W2 = 0
        self.dL_b2 = 0


    def _get_params(self):
        return(zip([self.W1, self.W2, self.b1, self.b2],
                   [self.dL_W1, self.dL_W2, self.dL_b1, self.dL_b2]))


    def _init_W(self, n_in, n_out):
        """initializes weight matrix using glorot initialization"""
        # Sample weights uniformly from [-1/sqrt(in), 1/np.sqrt(in)].
        bound = 1/np.sqrt(n_in)
        return(np.random.uniform(low=-bound, high=bound, size=(n_in, n_out)))


    def _get_minibatches(self, X, shuffle=True):
        """use the number of samples in X and k to determine the batch size"""
        idx = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(idx)

        # if the final batch size is smaller than k, append -1s for reshape
        if self.k > 1 and self.k != X.shape[0]:
            rem = len(idx) % self.k
            idx = np.hstack((idx, np.repeat(-1, (self.k-rem))))

        idx = idx.reshape(int(len(idx) / self.k), self.k)

        # load minibatches as a list of numpy arrays
        mbs = []
        for mb in np.arange(idx.shape[0]):
            mbs.append(idx[mb, :])

        # if -1's are in the final batch, remove them
        mbs[-1] = np.delete(mbs[-1], np.where(mbs[-1] == -1)[0])
        if len(mbs[-1]) == 0:
            mbs = mbs[:-1]

        return(mbs)


    def _softmax(self, X):
        """numerically stable softmax"""
        exps = np.exp(X - np.max(X, axis=1).reshape(-1, 1))
        return(exps / np.sum(exps, axis=1).reshape(-1, 1))


    def _softmax_backward(self, y_hat, y):
        """backprop of softmax"""
        y_one = self.onehot.fit_transform(y.reshape(-1,1))
        return(y_hat - y_one)


    def _relu_backward(self, z):
        """if previous layer pre-activated vals were <= 0, also set z to 0"""
        z[z <= 0] = 0
        z[z > 0] = 1
        return(z)


    def _relu(self, n):
        """calculates rectifier activation for all elements of n"""
        return(np.maximum(np.zeros(n.shape), n))


    def _nll_loss(self, x, y):
        y_hat = self.fprop(x, train=False)
        y_one = self.onehot.fit_transform(y.reshape(-1,1))

        prob = np.einsum('ky,ky->k', y_hat, y_one)
        loss = -np.log(prob)

        # add regularization
        if any ([self.l1, self.l2]):
            loss += self.l1 * np.sum(np.abs(self.W1))
            loss += self.l2 * np.sum(np.square(self.W1))
            loss += self.l1 * np.sum(np.abs(self.W2))
            loss += self.l2 * np.sum(np.square(self.W2))

        loss = np.mean(loss)

        return(loss)


    def fprop(self, X, train=True):
        """take inputs, and push them through to produce a prediction y"""

        if self.stupid_loop:

            all_ha, all_hs, all_oa, all_os = [], [], [], []

            for x in X:
                x = x.reshape(1,-1)  # reshape so that dimensions are kosher

                ha_sing = x.dot(self.W1) + self.b1;      all_ha.append(ha_sing)
                hs_sing= self._relu(ha_sing);            all_hs.append(hs_sing)
                oa_sing= hs_sing.dot(self.W2) + self.b2; all_oa.append(oa_sing)
                os_sing = self._softmax(oa_sing);        all_os.append(os_sing)

            ha = np.vstack(all_ha) # stack for compatibility with non-loop
            hs = np.vstack(all_hs) #
            oa = np.vstack(all_oa) #
            os = np.vstack(all_os) #

        else:
            ha = X.dot(self.W1) + self.b1
            hs = self._relu(ha)
            oa = hs.dot(self.W2) + self.b2
            os = self._softmax(oa) # this is y_hat

        assert(ha.shape == (X.shape[0], self.W1.shape[1]))
        assert(os.shape == (hs.shape[0], self.W2.shape[1]))

        LOGGER.debug('fprop 1 -- ha {} :: X.dot(W1) = {} . {}'.format(
            ha.shape, X.shape, self.W1.shape))
        LOGGER.debug('fprop 2 -- oa {} :: hs.dot(W2) = {} . {}'.format(
            os.shape, hs.shape, self.W2.shape))

        if train == True:
            self.ha, self.hs, self.oa, self.os = ha, hs, oa, os

        return(os)


    def bprop(self, X, y_hat, y, train=True):
        """
        backpropogate error between y_hat and y to update all parameters
        dL_b and dL_W are both normalized by batch size
        """
        if self.stupid_loop:

            all_dL_W1, all_dL_b1, all_dL_W2, all_dL_b2 = [], [], [], []

            y = y.reshape(-1,1) # used for k=1 case
            for i, (x, y_hat_1, y_1) in enumerate(zip(X, y_hat, y)):

                x = x.reshape(1,-1)  # reshape so that dimensions are kosher

                # gradients for output --> hidden layer
                dL_oa = self._softmax_backward(y_hat_1, y_1)
                dL_hs = self.W2.dot(dL_oa.T)
                hs_sing = self.hs[i, :].reshape(1, -1)
                dL_W2_sing = hs_sing.T.dot(dL_oa)
                dL_b2_sing = dL_oa

                all_dL_W2.append(dL_W2_sing)
                all_dL_b2.append(dL_b2_sing)

                # gradients for hidden --> input layer
                ha_sing = self.ha[i, :].reshape(1, -1)
                dhs_ha = self._relu_backward(ha_sing)
                dL_ha  = dhs_ha * dL_hs.T
                dL_W1_sing = x.T.dot(dL_ha)
                dL_b1_sing = dL_ha

                all_dL_W1.append(dL_W1_sing)
                all_dL_b1.append(dL_b1_sing)

            # NB: instead of dividing by self.this_k, just take the mean
            self.dL_W2 = np.mean(np.stack(all_dL_W2, axis=2), axis=2)
            self.dL_b2 = np.mean(np.stack(all_dL_b2, axis=2), axis=2).ravel()
            self.dL_W1 = np.mean(np.stack(all_dL_W1, axis=2), axis=2)
            self.dL_b1 = np.mean(np.stack(all_dL_b1, axis=2), axis=2).ravel()

        else:
            # NB: divide gradients by self.this_k here!
            # gradients for output --> hidden layer
            dL_oa = self._softmax_backward(y_hat, y)          # act wrt err
            dL_hs = self.W2.dot(dL_oa.T)                      # prev_activations
            self.dL_W2 = self.hs.T.dot(dL_oa) / self.this_k   # weights wrt err
            self.dL_b2 = np.sum(dL_oa, axis=0) / self.this_k  # bias wrt error

            # gradients for hidden --> input layer
            dhs_ha = self._relu_backward(self.ha)             # act wrt error
            dL_ha  = dhs_ha * dL_hs.T                         # pre/post act
            self.dL_W1  = X.T.dot(dL_ha) / self.this_k        # weights wrt err
            self.dL_b1  = np.sum(dL_ha, axis=0) / self.this_k # bias wrt err

        assert(self.dL_W2.shape == self.W2.shape)
        assert(self.dL_W1.shape == self.W1.shape)
        assert(self.dL_b1.shape == self.b1.shape)
        assert(self.dL_b2.shape == self.b2.shape)

        LOGGER.debug('bprop 2 -- dL_hs {} :: W2.dot(dL_oa) = {} . {}'.format(
            dL_hs.shape, self.W2.shape, dL_oa.T.shape))
        LOGGER.debug('bprop 2 -- dL_W2 {} :: hs.T.dot(dL_oa) = {} . {}'.format(
            self.dL_W2.shape, self.hs.T.shape, dL_oa.shape))
        LOGGER.debug('bprop 1 -- dL_ha {} :: dhs_ha*dL_hs.T = {} * {}'.format(
            dL_ha.shape, dhs_ha.shape, dL_hs.T.shape))
        LOGGER.debug('bprop 1 -- dL_W1 {} :: X.T.dot(dL_ha) = {} . {}'.format(
            self.dL_W2.shape, X.T.shape, dL_ha.shape))

        # calculate regularization
        reg_l11 = self.l1 * np.sign(self.W1)
        reg_l21 = self.l1 * np.sign(self.W2)
        reg_l12 = self.l2 * 2 * self.W1
        reg_l22 = self.l2 * 2 * self.W2

        # update all parameters via gradient descent with regularization
        self.W1 -= self.lr * (self.dL_W1 + reg_l11 + reg_l12)
        self.W2 -= self.lr * (self.dL_W2 + reg_l21 + reg_l22)
        self.b1 -= self.lr *  self.dL_b1
        self.b2 -= self.lr *  self.dL_b2


    def predict(self, X):
        y_hat = self.fprop(X, train=False)
        return(np.argmax(y_hat.T, axis=0))


    def accuracy(self, X, y):
        correct, total = 0, 0
        y_hat = self.predict(X)
        correct += (y_hat == y).sum()
        total += len(y)
        acc = correct / total

        return(round(acc*100, 4))


    def plot_decision(self, X, y, ax=None, h=0.07):
        """plot the decision boundary. h controls plot quality."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        # https://stackoverflow.com/a/19055059/6027071
        # sample a region larger than our training data X
        x_min = X[:, 0].min() - 0.5
        x_max = X[:, 0].max() + 0.5
        y_min = X[:, 1].min() - 0.5
        y_max = X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # plot decision boundaries
        x = np.concatenate(([xx.ravel()], [yy.ravel()]))
        pred = self.predict(x.T).reshape(xx.shape)
        ax.contourf(xx, yy, pred, alpha=0.8,cmap='RdYlBu')

        # plot points (coloured by class)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, cmap='RdYlBu')
        ax.axis('off')

        title = "nh={}_lr={}_l1={}_l2={}_epochs={}".format(
            self.n_h, self.lr, self.l1, self.l2, self.epochs)
        ax.set_title(title)

        output = 'decision_boundary_nh-{}_lr-{}_l1-{}_l2-{}_epochs-{}.jpg'.format(
            self.n_h, self.lr, self.l1, self.l2, self.epochs)
        plt.savefig(output)
        plt.close()


    def grad_check(self, X, y, name):
        """
        finite differences to check for correct gradient computation for each
        scalar parameter theta_k ,change the parameter value by adding a small
        perturbation (10^-5 and calculate the new value of the loss, then set
        the value of the parameter back to its original value. The partial
        derivative with respect to this parameter is estimated by dividing the
        change in the loss function by the pertubation. The ratio of your
        gradient computed by backpropagation and your estimate using finite
        difference should be between 0.99 and 1.01.
        """
        X = np.atleast_2d(X)     # order is crucial for k=1 case
        self.this_k = X.shape[0] #
        smol_eps = sys.float_info.min*sys.float_info.epsilon

        # initialized all parameters with some values
        y_hat = self.fprop(X, train=True)
        self.bprop(X, y_hat, y)

        loss_raw = self._nll_loss(X, y)

        grad_fds = []
        grad_bprops = []

        for param, grad in self._get_params():
            param = np.atleast_2d(param)
            grad = np.atleast_2d(grad)

            for theta_i in range(param.shape[0]):
                for theta_j in range(param.shape[1]):

                    eps = np.random.uniform(10e-6, 11e-4)
                    param[theta_i, theta_j] += eps
                    loss_mod = self._nll_loss(X, y)
                    param[theta_i, theta_j] -= eps

                    # do NOT normalize finite difference gradient by batch size!
                    # grad_bprop normalized by batch size in bprop function
                    grad_fd = (loss_mod - loss_raw) / (eps)
                    grad_bprop = grad[theta_i, theta_j]

                    # handle some edge cases, add smol_eps for stability
                    if np.abs(grad_fd - grad_bprop) < 10e-4:
                        ratio = 1
                        LOGGER.debug('almost no difference between grads, set to 1')
                    elif grad_bprop == 0:
                        ratio = 1
                        LOGGER.debug('grad_bprop = 0, set ratio to 1')
                    elif grad_fd == 0:
                        LOGGER.debug('grad_fd = 0, set ratio to 1')
                    else:
                        ratio = (grad_fd + smol_eps) / (grad_bprop + smol_eps)

                    grad_fds.append(grad_fd)
                    grad_bprops.append(grad_bprop)

                    if ratio < 0.99 or ratio > 1.01:
                        LOGGER.debug('gradient checking failed: loss_raw={}, loss_mod={}, fd={}, bprop={}'.format(
                            loss_raw, loss_mod, grad_fd, grad_bprop))
                        raise Exception('check ur grads bra')

        plt.plot(grad_fds, linewidth=2, alpha=0.7, color='r')
        plt.plot(grad_bprops, linewidth=1.8, alpha=0.7, linestyle='--', color='black')
        plt.legend(['finite differences', 'backpropagation'])
        plt.xlabel('parameter number')
        plt.ylabel('gradient')
        plt.savefig('gradient_differences_{}_k{}.jpg'.format(name, self.this_k))
        plt.close()


    def eval(self, X, y):

        if self.minibatch == False:
           self.k = X.shape[0]

        # split data into minibatches (incl. stochastic and batch case)
        minibatches = self._get_minibatches(X)

        total_acc = 0
        total_loss = 0

        for j, batch in enumerate(minibatches):
            total_acc += self.accuracy(X[batch, :], y[batch])
            total_loss += self._nll_loss(X[batch, :], y[batch])

        total_acc /= len(minibatches)
        total_loss /= len(minibatches)

        return(total_acc, total_loss)


    def train(self, data):

        results = {'loss':{'train': [], 'valid' : [], 'test': []},
                   'accuracy': {'train': [], 'valid': [], 'test': []}
        }

        X = data['X']['train']
        y = data['y']['train']

        for i in range(self.epochs):

            # batch gradient descent
            if self.minibatch == False:
               self.k = X.shape[0]

            # split data into minibatches (incl. stochastic and batch case)
            minibatches = self._get_minibatches(X)

            for j, batch in enumerate(minibatches):
                self.this_k = len(batch)
                LOGGER.debug('this_k = {}'.format(self.this_k))
                y_hat = self.fprop(X[batch, :], train=True)
                self.bprop(X[batch, :], y_hat, y[batch])

                acc_train, loss_train = self.eval(X, y)
                if j % 25 == 0:
                    LOGGER.info('TRAIN [epoch {}: batch {}/{}]: accuracy={}, loss={}'.format(
                        i+1, j+1, len(minibatches), acc_train, loss_train))

            # train accuracy & loss per epoch
            acc_valid, loss_valid = self.eval(data['X']['valid'], data['y']['valid'])
            acc_test, loss_test = self.eval(data['X']['test'], data['y']['test'])

            LOGGER.info('VALID [epoch {}: batch {}/{}]: accuracy={}, loss={}'.format(
                i+1, j+1, len(minibatches), acc_valid, loss_valid))
            LOGGER.info('TEST  [epoch {}: batch {}/{}]: accuracy={}, loss={}'.format(
                i+1, j+1, len(minibatches), acc_test, loss_test))

            # store end-of-epoch results for train, valid, test
            results['loss']['train'].append(loss_train)
            results['loss']['valid'].append(loss_valid)
            results['loss']['test'].append(loss_test)
            results['accuracy']['train'].append(acc_train)
            results['accuracy']['valid'].append(acc_valid)
            results['accuracy']['test'].append(acc_test)

        return(results)


def exp1():
    """
    As a beginning, start with an implementation that computes the gradients
    for a single example, and check that the gradient is correct using the
    finite difference method described above. Display the gradients for both
    methods (direct computation and finite difference) for a small network (e.g.
    d = 2 and d_h = 2) with random weights and for a single example.
    """
    data = get_circles_data()
    mlp = MLP(n_i=2, n_h=2, n_o=2, lr=5e-6, epochs=1, k=1, stupid_loop=True)
    #mlp.train(data)
    mlp.grad_check(data['X']['train'][100, :], data['y']['train'][100], 'first-check')


def exp2():
    """
    Add a hyperparameter for the minibatch size K to allow compute the
    gradients on a minibatch of K examples (in a matrix), by looping over
    the K examples (this is a small addition to your previous code).

    Display the gradients for both methods (direct computation and finite
    difference) for a small network (e.g. d = 2 and d h = 2) with random
    weights and for a minibatch with 10 examples (you can use examples from
    both classes from the two circles dataset).
    """
    data = get_circles_data()
    mlp = MLP(n_i=2, n_h=2, n_o=2, lr=5e-6, epochs=1, k=10, stupid_loop=True)
    #mlp.train(data)
    mlp.grad_check(data['X']['train'][:10, :], data['y']['train'][:10], 'loop-added')


def exp3():
    """
    Train your neural network using gradient descent on the two circles data-
    set. Plot the decision regions for several different values of the hyperpa-
    rameters (weight decay, number of hidden units, early stopping) so as to
    illustrate their effect on the capacity of the model.
    """
    data = get_circles_data()

    n_hs = [5, 100]
    lrs = [5e-2, 5e-3]
    l1s = [0, 0.01]
    l2s = [0, 0.01]
    epochs = [25, 100]

    # a stupid way to do grid search
    for n_h in n_hs:
        for lr in lrs:
            for l1 in l1s:
                for l2 in l2s:
                    for epoch in epochs:
                        mlp = MLP(n_i=2, n_o=2, n_h=n_h, lr=lr, l1=l1, l2=l2, epochs=epoch, k=100)
                        results = mlp.train(data)
                        mlp.plot_decision(data['X']['train'], data['y']['train'])


def exp4():
    """
    Compare both implementations (with a loop and with matrix calculus)
    to check that they both give the same values for the gradients on the
    parameters, first for K = 1, then for K = 10. Display the gradients for
    both methods.
    """
    data = get_circles_data()

    mlp = MLP(n_i=2, n_h=10, n_o=2, lr=5e-6, epochs=1, k=1)
    #results = mlp.train(data)
    mlp.grad_check(data['X']['train'][1, :], data['y']['train'][1], 'MATRIX')

    mlp = MLP(n_i=2, n_h=10, n_o=2, lr=5e-6, epochs=1, k=10)
    #results = mlp.train(data)
    mlp.grad_check(data['X']['train'][:10, :], data['y']['train'][:10], 'MATRIX')

    mlp = MLP(n_i=2, n_h=10, n_o=2, lr=5e-6, epochs=1, k=1, stupid_loop=True)
    #results = mlp.train(data)
    mlp.grad_check(data['X']['train'][1, :], data['y']['train'][1], 'LOOP')

    mlp = MLP(n_i=2, n_h=10, n_o=2, lr=5e-6, epochs=1, k=10, stupid_loop=True)
    #results = mlp.train(data)
    mlp.grad_check(data['X']['train'][:10, :], data['y']['train'][:10], 'LOOP')

def exp5():
    """
    Time how long takes an epoch on fashion MNIST (1 epoch = 1 full tra-
    versal through the whole training set) for K = 100 for both versions (loop
    over a minibatch and matrix calculus).
    """
    make_mnist_proc('data/fashion_mnist.pkl')
    data = load_pickle('data/fashion_mnist.pkl')

    t = time.time()
    mlp = MLP(n_i=784, n_h=10, n_o=10, lr=10e-03, epochs=1, k=100)
    results = mlp.train(data)
    elapsed_vector = time.time() - t

    t = time.time()
    mlp = MLP(n_i=784, n_h=10, n_o=10, lr=10e-03, epochs=1, k=100, stupid_loop=True)
    results = mlp.train(data)
    elapsed_loop = time.time() - t

    LOGGER.info('FASHION MNIST VECTOR time={}, LOOP time={}'.format(
        elapsed_vector, elapsed_loop))


def exp6():
    """
    Adapt your code to compute the error (proportion of misclassified examples)
    on the training set as well as the total loss on the training set during
    each epoch of the training procedure, and at the end of each epoch, it
    computes the error and average loss on the validation set and the test set.
    Display the 6 corresponding figures (error and average loss on
    train/valid/test), and write them in a log file. Train your network on the
    fashion MNIST dataset. Plot the training/valid/test curves (error and loss
    as a function of the epoch number, corresponding to what you wrote in a file
    in the last question). Add to your report the curves obtained using your
    best hyperparameters, i.e. for which you obtained your best error on the
    validation set. We suggest 2 plots : the first one will plot the error rate
    (train/valid/test with different colors, show which color in a legend) and
    the other one for the averaged loss (on train/valid/test). You should be
    able to get less than 20% test error.
    """
    data = load_pickle('data/fashion_mnist.pkl')
    options = {'k': 256, 'l11': 0, 'l12': 0.001, 'l21': 0, 'l22': 0.001}
    mlp = MLP(n_i=784, n_h=50, n_o=10, lr=10e-03, epochs=100, **options)
    results = mlp.train(data)

    i = 0
    plt.figure(1)
    plots = [231, 232, 233, 234, 235, 236]
    for result_type in results.keys():
        for trial_type in results[result_type].keys():
            plt.subplot(plots[i])
            plt.plot(results[result_type][trial_type],
                label='{} {}'.format(result_type, trial_type))
            plt.ylabel(result_type)
            plt.xlabel('epoch')
            plt.title(trial_type)
            i += 1

    plt.tight_layout()
    plt.savefig('mnist_learning curves.jpg')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="count",
        help="increase output verbosity")
    args = argparser.parse_args()

    if args.verbose != None:
        LOGGER.setLevel(logging.DEBUG)

    exp1() # LOOP gradient checking, and plot for k=1
    exp2() # LOOP gradient checking, and plot for k=10
    #exp3() # LOOP 2 show decision boundary
    exp4() # gradient check compare LOOP and VECTOR implementation

    # turn on logging logging
    #log_fname = "mlp_log.txt"
    #if os.path.isfile(log_fname):
    #    os.remove(log_fname)
    #log_hdl = logging.FileHandler(log_fname)
    #log_hdl.setFormatter(logging.Formatter('%(message)s'))
    #LOGGER.addHandler(log_hdl)

    #exp5() # time LOOP vs VECTOR implementation, K=100
    #exp6() # plots of train/valid/test error on MNIST

