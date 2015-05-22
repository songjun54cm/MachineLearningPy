#!/usr/bin/env python
# coding: utf-8

import time
import cPickle as pickle
import numpy as np
import Layers
from neuralnetwork import NeuralNetwork
from utils import one_hot

def run():
    # Fetch data
    f1 = file('../data/mldata/mnist_data.pkl','rb')
    mnist = pickle.load(f1)
    f1.close()
    split = 60000
    X_train = np.reshape(mnist.data[:split], (-1,1,28,28))/255.0
    Y_train = mnist.target[:split]
    X_test = np.reshape(mnist.data[split:], (-1,1,28,28))/255.0
    Y_test = mnist.target[split:]
    n_classes = np.unique(Y_train).size

    # Downsample training data
    n_train_samples = 3000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    Y_train = Y_train[train_idxs, ...]
    Y_train_one_hot = one_hot(Y_train)

    print ('number of train samples: %d')%(n_train_samples)
    print ('number of test samples: %d')%(X_test.shape[0])

    # setup network
    nn = NeuralNetwork(
        layers = [
            Layers.Convolution(
                n_feats=12, 
                filter_shape=(5,5),
                strides=(1,1),
                weight_scale=0.1,
                weight_decay=0.001),
            Layers.Activation('relu'),
            Layers.Pool(
                pool_shape=(2,2),
                strides=(2,2),
                mode='max'),
            Layers.Convolution(
                n_feats=16,
                filter_shape=(5,5),
                strides=(1,1),
                weight_scale=0.1,
                weight_decay=0.001),
            Layers.Activation('relu'),
            Layers.Flatten(),
            Layers.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02),
            Layers.Softmax()
            ]
        )

    #check gradient
    # nn.check_gradients(X_train[:10], Y_train_one_hot[:10])

    # Train neural network
    t0 = time.time()
    nn.train(X_train, Y_train_one_hot, learning_rate=0.05, max_iter=3, batch_size=32)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    # Y_test_one_hot = one_hot(Y_test)
    error = nn.error(X_test, Y_test)
    print('Test error rate: %.4f' % error)

if __name__=='__main__':
    run()