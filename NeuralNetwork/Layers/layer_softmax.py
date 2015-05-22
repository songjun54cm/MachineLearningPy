import numpy as np
from layer_example import Layer, LossMixin

class Softmax(Layer, LossMixin):
    """ SoftMmax layer with cross-entropy loss function. """
    def fprop(self, input):
        e = np.exp(input - np.amax(input, axis=1, keepdims=True))
        return e/np.sum(e, axis=1, keepdims=True)

    def bprop(self, output_grad):
        raise NotImplementedError(
                "Softmax layer does not support back-propagation of gradients." 
                + "It should occur only as the last layer of a NeuralNetwork.")

    def input_grad(self, Y, Y_pred):
        # Assumes one-hot encoding
        return (Y_pred - Y)/Y.shape[0]

    def loss(self, Y, Y_pred):
        # Aseumes Y is one-hot encoding.
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1-eps)
        Y_pred /= Y_pred.sum(axis=1, keepdims=True)
        loss = -np.sum(Y * np.log(Y_pred))
        return loss / Y.shape[0]

    def get_output_shape(self, input_shape):
        return input_shape