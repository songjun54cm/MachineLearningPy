import numpy as np
from scipy.spatial.distance import cdist
import copy
from model_basic import ModelBasic

class KNN_Model(ModelBasic):
    def __init__(self, params):
        self.model_name = 'KNN'
        self.K = params['K']
        self.debug = params['debug']
        print 'KNN model inited.'

    def train(self, train_data, train_labels):
        # train the model
        # train_data, NxF data matrix with one data sample per row
        # train_label, N, data labels with one label per element
        self.train_data = copy.deepcopy(train_data)
        self.train_labels = copy.deepcopy(train_labels)

    def predict(self, test_data):
        # predict the label of test_data
        # test_data, NxF data matrix with one data sample per row
        pred_labels = []
        diss =cdist(test_data, self.train_data)
        near_idx = np.argsort(diss,axis=1)[:,0:self.K]
        for idx in near_idx:
            max_num = 0
            max_key = -1
            cand_key = {}
            for i in idx:
                temp_key = self.train_labels[i]
                if cand_key.has_key(temp_key):
                    cand_key[temp_key] += 1
                else:
                    cand_key[temp_key] = 1
                if cand_key[temp_key]>=max_num:
                    max_key = temp_key
                    max_num = cand_key[temp_key]

            pred_labels.append(max_key)
        return pred_labels

    def test(self, test_data, test_labels):
        pred_labels = self.predict(test_data)
        num_total = len(pred_labels)
        num_correct = 0
        for i,label in enumerate(test_labels):
            if label == pred_labels[i]:
                num_correct += 1
        print 'accuracy: %f\n' % (num_correct/num_total)

def get_train_data():
    train_data = np.array([[1.0, 1.1],[1.0, 1.0],[0.0,0.0],[0.0,0.1]])
    train_labels = ['A', 'A', 'B', 'B']
    return train_data, train_labels

def get_test_data():
    test_data = np.array([[1.1, 1.0],[1.1, 0.9],[0.1,-0.1],[0.1,0.0]])
    test_labels = ['A', 'A', 'B', 'B']
    return test_data, test_labels

def test():
    train_data, train_labels = get_train_data()
    test_data, test_labels = get_test_data()
    params = {}
    params['K'] = 1
    params['debug'] = False
    knn = KNN_Model(params)
    knn.train(train_data, train_labels)
    knn.test(test_data, test_labels)

if __name__=='__main__':
    test()
