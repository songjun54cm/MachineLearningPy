import numpy as np

class Layer(object):
    def _setup(self, input_shape, rng):
        """setup layer with parameters that are unknown at __init__()"""
        pass
	
    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def get_output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()

    def display(self):
        print self.__class__.__name__

class LossMixin(object):
    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()

class ParamMixin(object):
    def params(self):
        """ Layer parameters. """
        raise NotImplementedError()

    def param_grads(self):
        """ Get layer parameter gradients as calculated from bprop(). """
        raise NotImplementedError()

    def param_incs(self):
        """ Get layer parameter steps as calculated from bprop(). """
        raise NotImplementedError()

    def update_params(self, learning_rate):
        """ Update Parameters. """
        raise NotImplementedError()

def compare_gradient(costFunc, layer, params, Y):
    # check params gradients
    assert len(params.shape)==1, 'params must be raveled to one dimensional vector'
    cost, grad = costFunc(layer, params, Y)
    # params_grad = np.zeros(theta.shape)
    mu = 1e-6
    param_max_diff = 0
    param_max_pos = 0
    for i in range(params.shape[0]):
        params[i] += mu
        new_cost, new_grad = costFunc(layer, params, Y)
        numer_grad = (new_cost-cost)/mu
        params[i] -= mu
        curr_diff = np.abs(grad[i]-numer_grad)/np.abs(numer_grad)
        print('(%d/%d) param_grad: %.4f, numer_grad: %.4f curr_diff: %.4f'
            % (i,params.shape[0],grad[i], numer_grad, curr_diff))
        if curr_diff > param_max_diff:
            param_max_pos = i
            param_max_diff = curr_diff
    return param_max_diff, param_max_pos