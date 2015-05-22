import numpy as np
from layer_example import Layer

class Activation(Layer):
    def __init__(self, type):
        if type=='sigmod':
            self.fun = self.sigmoid
            self.fun_d = self.sigmoid_d
        elif type == 'relu':
            self.fun = self.relu
            self.fun_d = self.relu_d
        elif type == 'tanh':
            self.fun = self.tanh
            self.fun_d = self.tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_d(self, x):
        s = self.sigmoid(x)
        return s*(1.0-s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        e = np.exp(2*x)
        return (e-1)/(e+1)

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_d(self, x):
        dx = np.zeros(x.shape)
        dx[x >= 0] = 1
        return dx

    def fprop(self, input_data):
        self.last_input_data = input_data
        return self.fun(input_data)
    def bprop(self, output_grad):
        return output_grad * self.fun_d(self.last_input_data)

    def get_output_shape(self, input_shape):
        return input_shape