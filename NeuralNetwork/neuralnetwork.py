import numpy as np
import scipy as sp
from Layers import ParamMixin
from utils import one_hot, unhot

class NeuralNetwork:
    def __init__(self, layers, rng=None):
        self.layers = layers
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def _setup(self, X, Y):
        # Setup layers sequentially
        input_shape = X.shape
        for layer in self.layers:
            layer._setup(input_shape, self.rng)
            input_shape = layer.get_output_shape(input_shape)
        if input_shape != Y.shape:
            raise ValueError("Output shape %s does not match Y %s" % 
                    (input_shape, Y.shape))

    def forward(self, X_batch):
        # Forward propagation
        X_next = X_batch
        for layer in self.layers:
            X_next = layer.fprop(X_next)
        return X_next

    def backward(self, Y_batch, Y_pred, learning_rate):
        next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
        for layer in reversed(self.layers[:-1]):
            next_grad = layer.bprop(next_grad)
            # update parameters
            if isinstance(layer, ParamMixin):
                layer.update_params(learning_rate)
                        
    def train(self, X, Y, learning_rate=0.05, max_iter=3, batch_size=32):
        """Train network on the given data."""
        print ('train network with \n \
            learning_rate: %.4f, \n \
            max_iter: %d, \n \
            batch_size: %d \n' \
            %(learning_rate, max_iter, batch_size))
        
        num_sample = Y.shape[0]
        num_batch = num_sample//batch_size
        # Y_one_hot = one_hot(Y)
        self._setup(X, Y)
        self.display()
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(num_batch):
                batch_begin = b*batch_size
                batch_end = batch_begin + batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y[batch_begin:batch_end]

                # Forward propagation
                Y_pred = self.forward(X_batch)

                # Back propagation of partial derivatives
                self.backward(Y_batch, Y_pred, learning_rate)

            # Output training status
            loss = self._loss(X, Y)
            # loss = self.layers[-1].loss(Y_batch, Y_pred)
            error = self.error(X, unhot(Y))
            # error = unhot(Y_pred) != Y
            # error = np.mean(error)
            print("iter %i, loss %.4f, train error %.4f' " % (iter, loss, error))

    def _loss(self, X, Y_one_hot):
        Y_pred = self.forward(X)
        return self.layers[-1].loss(Y_one_hot,Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        Y_pred = self.forward(X)
        Y_pred = unhot(Y_pred)
        return Y_pred

    def error(self, X, Y):
        Y_pred = self.predict(X)
        error = Y != Y_pred
        return np.mean(error)

    def display(self):
        for l in self.layers:
            l.display()

    def check_gradients(self, X, Y_one_hot):
        self._setup(X, Y_one_hot)
        for l, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                print('layer %d' % l)
                for p, param in enumerate(layer.params()):
                    param_shape = param.shape
                    def fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        return self._loss(X, Y_one_hot)
                    def grad_fun(param_new):
                        param[:] = np.reshape(param_new, param_shape)
                        X_next = X
                        for layer in self.layers:
                            X_next = layer.fprop(X_next)
                        Y_pred = X_next

                        next_grad = self.layers[-1].input_grad(Y_one_hot,Y_pred)
                        for layer in reversed(self.layers[l:-1]):
                            next_grad = layer.bprop(next_grad)
                        return np.ravel(self.layers[l].param_grads()[0])

                    param_init = np.ravel(np.copy(param))
                    err = sp.optimize.check_grad(fun, grad_fun, param_init)
                    print('diff %.2e' % err)