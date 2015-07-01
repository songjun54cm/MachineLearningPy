class ModelBasic(object):
    def __init__(self, params):
        raise NotImplementedError()

    def train(self, train_data, train_labels):
        raise NotImplementedError()

    def predict(self, test_data):
        raise NotImplementedError()

    def test(self, test_data, test_labels):
        raise NotImplementedError()