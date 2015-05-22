import numpy as np
from scipy import signal
from layer_example import Layer, ParamMixin
import sys

class Convolution(Layer, ParamMixin):
    def __init__(self, n_feats, filter_shape, strides=(1,1), weight_scale=1,
                weight_decay=0.0, padding_mode='valid', border_mode='nearest'):
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay
        self.padding_mode = padding_mode
        self.border_mode = border_mode

    def _setup(self, input_shape, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        n_channels = input_shape[1]
        W_shape = (n_channels, self.n_feats) + self.filter_shape
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_feats)
        self.param_shapes = (self.W.shape, self.b.shape)
        self.output_shape = self.get_output_shape(input_shape)

    def fprop(self, input_data):
        #TODO padding_mode 'same' and 'full'
        #TODO strides unequals to 1
        self.last_input = input_data
        
        # print self.output_shape, input_data.shape, self.W.shape
        convout = np.zeros(self.get_output_shape(input_data.shape))
        for n in range(convout.shape[0]):
            for f in range(convout.shape[1]):
                for c in range(input_data.shape[1]):
                    try:
                        convout[n, f, :, :] += signal.convolve2d(input_data[n,c,:,:], np.rot90(np.rot90(self.W[c,f,:,:])), mode=self.padding_mode)
                    except:
                        print 'error'
        return convout + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    def bprop(self, output_grad):
        #TODO padding_mode 'same' and 'full'
        if self.padding_mode == 'valid':
            input_bp_mode = 'full'
            param_bp_mode = 'valid'
            padding_input_data = self.last_input
        elif self.padding_mode == 'same':
            input_bp_mode = 'same'
            param_bp_mode = 'valid'
            padding_size = self.W.shape[2]//2
            padding_input_data = np.zeros((self.last_input.shape[0], 
                                            self.last_input.shape[1],
                                            self.last_input.shape[2]+self.W.shape[2],
                                            self.last_input.shape[3]+self.W.shape[3]))
            padding_input_data[:,:,
                                padding_size:self.last_input.shape[2]+padding_size,
                                padding_size:self.last_input.shape[3]+padding_size] = self.last_input
        input_grad = np.zeros(self.last_input.shape)
        self.dW = np.zeros(self.W.shape)
        for n in range(output_grad.shape[0]):
            for f in range(output_grad.shape[1]):
                for c in range(self.last_input.shape[1]):
                    input_grad[n, c, :, :] += signal.convolve2d(output_grad[n,f,:,:], self.W[c,f,:,:], mode=input_bp_mode)
                    self.dW[c,f,:,:] += signal.convolve2d(padding_input_data[n,c,:,:], np.rot90(np.rot90(output_grad[n,f,:,:])), mode=param_bp_mode)
        self.db = np.sum(output_grad, axis=(0,2,3))
        self.dW -= self.weight_decay * self.W
        return input_grad

    def update_params(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def get_output_shape(self,input_shape):
        #TODO same and full
        if self.padding_mode == 'same':
            h = input_shape[2]
            w = input_shape[3]
        elif self.padding_mode == 'valid':
            h = input_shape[2]-self.filter_shape[0]+1
            w = input_shape[3]-self.filter_shape[1]+1
        elif self.padding_mode == 'full':
            h = input_shape[2]+self.filter_shape[0]-1
            w = input_shape[3]+self.filter_shape[1]-1
        return (input_shape[0], self.n_feats, h, w)

    def params(self): 
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay
        gW = self.dW + self.weight_decay*self.W
        return gW, self.db

    def set_params(self, new_params):
        self.W = new_params[0]
        self.b = new_params[1]

    def display(self):
        print self.__class__.__name__
        print '\tparam: W %s' % str(self.W.shape)
        print '\tparam: b %s' % str(self.b.shape)

def check_gradient():
    X = np.random.random((2,3,10,10))
    # X = np.reshape(np.arange(1,19,1),(2,1,3,3))/10
    # print X
    layer = Convolution(4,(3,3),1)
    layer._setup(X.shape)
    Y = np.random.random(layer.get_output_shape(X.shape))

    # Y = np.ones(layer.output_shape)/10


    param_W, param_b = layer.params()
    params = np.concatenate((np.ravel(param_W), np.ravel(param_b)),axis=1)

    def costFunc(layer, X, Y):
        grad = 0
        Y_pred = layer.fprop(X)
        cost = 0.5 * np.sum( ( Y-Y_pred ) **2 ) / Y.shape[0]
        output_grad = (Y_pred-Y)/Y.shape[0]
        layer.bprop(output_grad);
        grad = layer.param_grads()
        param_grad = np.concatenate((np.ravel(grad[0]) ,np.ravel(grad[1])), axis=1)
        return cost, param_grad

    def costFunc2(layer, X, Y):
        Y_pred = layer.fprop(X)
        cost = 0.5 * np.sum( ( Y-Y_pred ) **2 ) / Y.shape[0]
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

    cost, grad = costFunc(layer, X, Y)
    # params_grad = np.zeros(theta.shape)
    mu = 1e-6
    # check gradient of params
    para_max_diff = 0
    para_max_pos = 0
    for i in range(params.shape[0]):
        params[i] += mu
        set_new_params(layer, params)
        
        new_cost, new_grad = costFunc(layer, X, Y)
        numer_grad = (new_cost-cost)/mu
        params[i] -= mu
        curr_diff = np.abs(grad[i]-numer_grad)/np.abs(numer_grad)
        # print('cost: %.4f, new_cost: %.4f') % (cost, new_cost)
        print('(%d/%d) param_grad: %.4f, numer_grad: %.4f curr_diff: %.4e'
            % (i,params.shape[0], new_grad[i], numer_grad, curr_diff))
        if curr_diff > para_max_diff:
            para_max_pos = i
            para_max_diff = curr_diff

    # check gradient of input
    cost, input_grad = costFunc2(layer, X, Y)
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
        # print('cost: %.4f, new_cost: %.4f') % (cost, new_cost)
        print('(%d/%d) input_grad: %.4f, numer_grad: %.4f curr_diff: %.4e'
            % (i, input_x.shape[0], new_grad[i], numer_grad, curr_diff))
        if curr_diff > input_max_diff:
            input_max_pos = i
            input_max_diff = curr_diff
    print('para_max_diff: %.4e, para_max_pos: %.4f') % (para_max_diff, para_max_pos)
    print('input_max_diff: %.4e, input_max_pos: %.4f') % (input_max_diff, input_max_pos)
    # print Y
    # Y_pred = layer.fprop(X)
    # print Y_pred
if __name__=="__main__":
    if sys.argv[1]=='check_gradient':
        check_gradient()