from model_basic import ModelBasic
import numpy as np
from mlutils import majorityVote

def probabilities(x):
    # x: numpy.array(N), N samples
    var_set = list(set(list(x)))
    num_sample = float(x.shape[0])
    pros = np.zeros(len(var_set))
    for vi, var in enumerate(var_set):
        pros[vi] = np.sum(x == var)/num_sample
    return var_set, pros

def get_entropy(x):
    # x: np.array(N)
    _, probability = probabilities(x)
    entropy = -np.dot(probability,np.log(probability).T)
    return entropy

def conditional_entropy(y,x):
    # y: np.array(N), N sample with label
    # x: np.array(N), N variable
    var_set = list(set(list(x)))
    probability = np.zeros(len(var_set))
    entropies = np.zeros(len(var_set))
    num_sample = x.shape[0]
    for vi,var in enumerate(var_set):
        y_var = y[x==var]
        entropies[vi] = get_entropy(y_var)
        probability[vi] = np.sum(x==var)/num_sample
    cond_entr = np.dot(probability, entropies.T)
    return cond_entr, var_set

def get_train_data():
    train_data = np.array([[1,1],[1,1],[1,0],[0,1],[0,1]])
    train_labels = np.array(['y','y','n','n','n'])
    return train_data, train_labels

def get_test_data():
    test_data = np.array([[1,1],[1,0],[0,1]])
    test_labels = np.array(['y','n','n'])
    return test_data, test_labels

class DecisionTreeModel(ModelBasic):
    class Node:
        def __init__(self):
            self.fea_index = -1
            self.label_list = []
            self.label = None
            self.label_probabilities = []
            self.children_nodes = {}

        def predict(self, test_data):
            if self.fea_index < 0:
                return self.label
            else:
                return self.children_nodes[test_data[self.fea_index]].predict(test_data)

    def __init__(self, params=None):
        self.debug = params['debug']

    @staticmethod
    def select_feature(fea_x, label_y, fea_idxes):
        max_inc_en = 0 # max increase entropy
        max_ce_idx = -1 # max conditional entropy index
        selected_fea_var_set = None
        for idx in fea_idxes:
            en = get_entropy(label_y)
            ce, fea_var_set = conditional_entropy(label_y, fea_x[:,idx])
            inc_en = en - ce
            if inc_en >= max_inc_en:
                max_inc_en = inc_en
                max_ce_idx = idx
                selected_fea_var_set = fea_var_set
        return max_ce_idx, selected_fea_var_set

    @staticmethod
    def create_node(fea_x, label_y, fea_indi):
        cur_node = DecisionTreeModel.Node()
        #cur_node.label_list, cur_node.label_probabilities = probabilities(label_y)
        if np.sum(label_y==label_y[0]) == label_y.shape[0]:
            cur_node.label = label_y[0]
            return cur_node

        fea_idxes = fea_indi.nonzero()[0]
        if len(fea_idxes)==0:
            cur_node.label = majorityVote(label_y)
        else:
        # select feature
            fea_index, fea_var_set = DecisionTreeModel.select_feature(fea_x, label_y, fea_idxes)
            cur_node.fea_index = fea_index
            fea_indi[fea_index] = False
            for fea_var in fea_var_set:
                tmp_indi = (fea_x[:,fea_index]==fea_var).nonzero()[0]
                tmp_fea_x = fea_x[tmp_indi,:]
                tmp_label_y = label_y[tmp_indi]
                cur_node.children_nodes[fea_var] = DecisionTreeModel.create_node(tmp_fea_x, tmp_label_y, fea_indi)
        return cur_node

    @staticmethod
    def create_decision_tree(train_data, train_labels):
        fea_indi = np.ones(train_data.shape[1])>0
        root = DecisionTreeModel.create_node(train_data, train_labels, fea_indi)
        return root

    def train(self, train_data, train_labels):
        # train the model
        # train_data, NxF data matrix with one data sample per row
        # train_label, N, data labels with one label per element
        self.tree = self.create_decision_tree(train_data,train_labels)

    def predict(self, test_data):
        pred_labels = []
        for td in test_data:
            pred_labels.append(self.tree.predict(td))
        return pred_labels

    def test(self, test_data, test_labels):
        pred_labels = self.predict(test_data)
        num_total = len(pred_labels)
        num_correct = 0
        for i,label in enumerate(test_labels):
            if label == pred_labels[i]:
                num_correct += 1
        print 'accuracy: %f\n' % (num_correct/num_total)

def test():
    train_data, train_labels = get_train_data()
    test_data, test_labels = get_test_data()
    params = {}
    params['debug'] = False
    decision_tree = DecisionTreeModel(params)
    decision_tree.train(train_data, train_labels)
    decision_tree.test(test_data, test_labels)

if __name__=='__main__':
    test()