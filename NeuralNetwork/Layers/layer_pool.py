import sys
import numpy as np
from layer_example import Layer

class Pool(Layer):
    def __init__(self, pool_shape=(2,2), strides=(2,2), mode='max'):
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides

    def fprop(self, input_data):
        self.last_data = input_data
        self.last_switches = np.empty(self.get_output_shape(input_data.shape)+(2,), dtype=np.int)
        pool_out = np.zeros(self.get_output_shape(input_data.shape))

        pool_h_top = self.pool_h//2 - 1 + self.pool_h % 2 # if the hight of pool window is even, the center is near the left-top
        pool_h_bottom = self.pool_h//2+1
        pool_w_left = self.pool_w//2 - 1 + self.pool_w % 2 # if the width of pool window is even, the center is near the left-top
        pool_w_right = self.pool_w//2  + 1

        for n in range(pool_out.shape[0]):
            for f in range(pool_out.shape[1]):
                for y_out in range(pool_out.shape[2]):
                    y = y_out * self.stride_y
                    y_min = max(y-pool_h_top, 0)
                    y_max = min(y+pool_h_bottom, input_data.shape[2])
                    for x_out in range(pool_out.shape[3]):
                        x = x_out * self.stride_x
                        x_min = max(x-pool_w_left, 0)
                        x_max = min(x+pool_w_right, input_data.shape[3])
                        region = input_data[n,f,y_min:y_max, x_min:x_max]
                        if self.mode=='max':
                            max_0, argmax_0 = region.max(0), region.argmax(0)
                            max_1, argmax_1 = max_0.max(), max_0.argmax()
                            maxVal = max_1
                            max_pos_y, max_pos_x = argmax_0[argmax_1], argmax_1
                            pool_out[n,f,y_out,x_out] = maxVal
                            self.last_switches[n,f,y_out,x_out,0] = max_pos_y + y_min
                            self.last_switches[n,f,y_out,x_out,1] = max_pos_x + x_min
                            # print max_pos_y, max_pos_x
                        else:
                            raise ValueError('Error Pooling Mode')
        return pool_out

    def bprop(self, output_grad):
        input_grad = np.zeros(self.last_data.shape)
        for n in range(output_grad.shape[0]):
            for f in range(output_grad.shape[1]):
                for y_out in range(output_grad.shape[2]):
                    for x_out in range(output_grad.shape[3]):
                        input_grad[n,f,self.last_switches[n,f,y_out,x_out,0],self.last_switches[n,f,y_out,x_out,1]] \
                            = output_grad[n,f,y_out,x_out]

        return input_grad
    def get_output_shape(self, input_shape):
        shape = (input_shape[0],
                input_shape[1],
                #input_shape[2]//self.stride_y,
                #input_shape[3]//self.stride_x,
                np.ceil(float(input_shape[2])/self.stride_y),
                np.ceil(float(input_shape[3])/self.stride_x))
        return shape


def check_gradient():
    X = np.random.random((3,4,3,3))/10
    # print X
    layer = Pool()
    Y_pred = layer.fprop(X)
    # print Y_pred
    Y = np.random.random(layer.get_output_shape(X.shape))/10

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

    # check gradient of input
    mu = 1e-6
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
    # print('para_max_diff: %.4f, para_max_pos: %.4f') % (para_max_diff, para_max_pos)
    print('input_max_diff: %.4e, input_max_pos: %.4f') % (input_max_diff, input_max_pos)

if __name__=="__main__":
    if sys.argv[1]=='check_gradient':
        check_gradient()