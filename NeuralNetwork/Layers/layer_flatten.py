import sys
import numpy as np
from layer_example import Layer, compare_gradient

class Flatten(Layer):
    def fprop(self, input_data):
        self.last_input_shape = input_data.shape
        return np.reshape(input_data, (input_data.shape[0],-1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def get_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))


def check_gradient():
    X = np.random.random((3,5,5))
    Y = np.random.random((X.shape[0], np.prod((X.shape[1],X.shape[2]))))
    layer = Flatten()
    layer.last_input_shape = X.shape

    def costFunc_input(layer, params, Y):
        X = np.reshape(params, layer.last_input_shape)
        Y_pred = layer.fprop(X)
        cost = 0.5 * np.sum( ( Y-Y_pred ) **2)/Y.shape[0]
        output_grad = (Y_pred-Y)/Y.shape[0]
        input_grad = layer.bprop(output_grad);
        return cost, np.ravel(input_grad)

    params = np.ravel(X)
    # check input gradients
    input_max_diff, input_max_pos = compare_gradient(costFunc_input, layer, params, Y)
    print('input_max_diff: %.4e, input_max_pos: %.4f') % (input_max_diff, input_max_pos)

if __name__=="__main__":
    if sys.argv[1]=='check_gradient':
        check_gradient()