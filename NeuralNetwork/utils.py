import numpy as np

def one_hot(labels):
    classes = np.unique(labels)
    num_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (num_classes,))
    for c in classes:
        one_hot_labels[labels==c, c] = 1
    return one_hot_labels

def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)

def param2stack(params):
    num_params = len(params)
    param_stack = np.array()
    param_shape = []
    for i in range(num_params):
        param_shape.append(params[i].shape)
        para_s = np.ravel(np.copy(params[i]))
        np.concatenate((param_stack, para_s),axis=1)
    return param_stack, param_shape

def stack2param(param_stack, param_shape):
    num_params = len(param_shape)
    params = []
    len_start = 0
    len_end = 0
    for i in range(num_params):
        length = np.prod(param_shape[i])
        len_start = len_end
        len_end += length
        params.append(np.reshape(param_stack[len_start:len_end], param_shape))
    return params