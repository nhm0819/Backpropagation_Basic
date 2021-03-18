import numpy as np

class dataset_generator:
    def __init__(self, feature_dim = 1, n_sample=100, noise=0):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise = noise

        self._coefficient = None
        self._coefficient_list = None
        self._distribution_params = None

        self._dataset = None

        self.__init_set_coefficient()
        self._init_distribution_params()

    def __init_set_coefficient(self):
        self._coefficient = [1 for _ in range(self._feature_dim)] + [0]

    def _init_distribution_params(self):
        self._distribution_params = {f:{'mean':0, 'std':1}
                                     for f in range(1, self._feature_dim+1)}

    def set_n_sample(self, n_sample):
        self._n_sample = n_sample

    def set_noise(self, noise):
        self._noise = noise

    def set_coefficient(self, coefficient_list):
        self._coefficient = coefficient_list

    def set_distribution_params(self, distribution_params):
            self._distribution_params = param_value

    def make_dataset(self):
        x_data = np.random.normal(0, 1, size=(self._n_sample, self._feature_dim))

        y_data = np.zeros(shape=(self._n_sample, 1))
        for feature_idx in range(self._feature_dim):
            y_data += self._coefficient[feature_idx]*x_data[:,feature_idx].reshape(-1,1)
        y_data += self._coefficient[-1]
        y_data += self._noise*np.random.normal(0,1, size=(self._n_sample, 1))

        return x_data, y_data