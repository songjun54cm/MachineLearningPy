import numpy as np
from layer_example import Layer, ParamMixin
import sys

class Linear(Layer, ParamMixin):
    def __init__(self, n_out, weight_scale=1, weight_decay=0.0):
        self.n_out = n_out;
        self.weight_decay = weight_decay
        self.weight_scale = weight_scale

    def _setup(self, input_shape, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        n_input = input_shape[1]
        W_shape = (n_input, self.n_out)
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_out)
        self.param_shapes = (self.W.shape, self.b.shape)

    def fprop(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.W) + self.b

    def bprop(self, output_grad):
        # n = output_grad.shape[0]
        self.dW = np.dot(self.last_input.T, output_grad) - self.weight_decay*self.W
        self.db = np.sum(output_grad, axis=0)
        return np.dot(output_grad, self.W.T)

    def update_params(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def set_params(self, new_params):
        self.W = new_params[0]
        self.b = new_params[1]

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay
        gW = self.dW + self.weight_decay*self.W
        return gW, self.db

    def get_output_shape(self, input_shape):
        return (input_shape[0], self.n_out)

    def display(self):
        print self.__class__.__name__
        print '\tparam: W %s' % str(self.W.shape)
        print '\tparam: b %s' % str(self.b.shape)

def check_gradient():
    X = np.random.random((3,5))
    Y = np.random.random((3,10))
    layer = Linear(10)
    layer._setup(X.shape)

    param_W, param_b = layer.params()
    params = np.concatenate((np.ravel(param_W),np.ravel(param_b)),axis=1)

    def costFunc(layer, X, Y):
        Y_pred = layer.fprop(X)
        cost = 0.5 * np.sum( ( Y-Y_pred ) **2)/Y.shape[0]
        output_grad = (Y_pred-Y)/Y.shape[0]
        layer.bprop(output_grad);
        grad = layer.param_grads()
        grad = np.concatenate((np.ravel(grad[0]) ,np.ravel(grad[1])), axis=1)
        return cost, grad

    def costFunc2(layer, X, Y):
        Y_pred = layer.fprop(X)
        cost = 0.5 * np.sum( ( Y-Y_pred ) **2)/Y.shape[0]
        output_grad = (Y_pred-Y)/Y.shape[0]
        input_grad = layer.bprop(output_grad);
        return cost, np.ravel(input_grad)

    def set_new_params(layer, params):
        p_shape = layer.param_shapes
        pos_start = 0
        pos_end = 0
        new_paras = []
        for s in p_shape:
            pos_end =pos_start + np.prod(s)
            new_paras.append(np.reshape(params[pos_start:pos_end],s))
            pos_start = pos_end
        layer.set_params(new_paras)

    # check params gradients
    cost, grad = costFunc(layer, X, Y)
    # params_grad = np.zeros(theta.shape)
    mu = 1e-6
    param_max_diff = 0
    param_max_pos = 0
    for i in range(params.shape[0]):
        params[i] += mu
        set_new_params(layer, params)
        new_cost, new_grad = costFunc(layer, X, Y)
        numer_grad = (new_cost-cost)/mu
        params[i] -= mu
        curr_diff = np.abs(grad[i]-numer_grad)/np.abs(numer_grad)
        print('(%d/%d) param_grad: %.4f, numer_grad: %.4f curr_diff: %.4e'
            % (i,params.shape[0],grad[i], numer_grad, curr_diff))
        if curr_diff > param_max_diff:
            param_max_pos = i
            param_max_diff = curr_diff

    # check input gradients
    cost, input_grad = costFunc2(layer, X, Y)
    mu = 1e-6
    input_max_diff = 0
    input_max_pos = 0
    input_x = np.ravel(X)
    for i in range(input_x.shape[0]):
        input_x[i] += mu
        new_X = np.reshape(input_x, X.shape)
        new_cost, new_grad = costFunc2(layer, new_X, Y)
        numer_grad = (new_cost-cost)/mu
        input_x[i] -= mu
        curr_diff = np.abs(input_grad[i]-numer_grad)/np.abs(numer_grad)
        print('(%d/%d) input_grad: %.4f, numer_grad: %.4f curr_diff: %.4e'
            % (i,input_x.shape[0],input_grad[i], numer_grad, curr_diff))
        if curr_diff > input_max_diff:
            input_max_pos = i
            input_max_diff = curr_diff
    print('param_max_diff: %.4e, param_max_pos: %.4f') % (param_max_diff, param_max_pos)
    print('input_max_diff: %.4e, input_max_pos: %.4f') % (input_max_diff, input_max_pos)

if __name__=="__main__":
    if sys.argv[1]=='check_gradient':
        check_gradient()

        