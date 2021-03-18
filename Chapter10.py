import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import basic_nodes as nodes

np.random.seed(0)

plt.style.use('seaborn')
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 15

class dataset_generator:
    def __init__(self, feature_dim, n_sample=300, noise_factor=0., direction=1):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise_factor = noise_factor
        self._direction = direction

        self._init_feature_dict()
        self._init_t_th()

    def _init_feature_dict(self):
        self._feature_dict = dict()
        for feature_idx in range(1, self._feature_dim+1):
            x_dict = {'mean':0, 'std':1}
            self._feature_dict[feature_idx] = x_dict

    def _init_t_th(self):
        self._t_th = [0] + [1 for i in range(self._feature_dim)]

    def set_feature_dict(self, feature_dict):
        if len(feature_dict) != self._feature_dim:
            class FeatureDictError(Exception):
                pass
            raise FeatureDictError('The Length of "feature_dict" should be equal to "feature_dim"')
        else:
            self._feature_dict = feature_dict

    def set_t_th(self, t_th_list):
        if len(t_th_list) != len(self._t_th):
            class t_th_Error(Exception):
                pass
            raise t_th_Error('The Length of "t_th_list" should be equal to "feature_dim + 1"')
        else:
            self._t_th = t_th_list

    def make_dataset(self):
        x_data = np.zeros(shape=(self._n_sample, 1))
        y = np.zeros(shape=(self._n_sample, 1))

        for feature_idx in range(1, self._feature_dim+1):
            feature_dict = self._feature_dict[feature_idx]
            data = np.random.normal(loc = feature_dict['mean'], scale = feature_dict['std'], size = (self._n_sample,1))

            x_data = np.hstack((x_data, data))
            y += self._t_th[feature_idx] * data

        y += self._t_th[0]
        y_noise = y + self._noise_factor*np.random.normal(0,1,(self._n_sample,1))

        if self._direction > 0:
            y_data = (y_noise > 0).astype(np.int)
        else:
            y_data = (y_noise < 0).astype(np.int)

        data = np.hstack((x_data, y_data))
        return data

def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx*batch_size:]
    else:
        batch = data[batch_idx*batch_size:(batch_idx+1)*batch_size]
    return batch


def dataset_visualizer():
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(projection = '3d')
    ax.plot(data[p_idx,1].flat, data[p_idx,2].flat, data[p_idx,-1].flat,'bo')
    ax.plot(data[np_idx,1].flat, data[np_idx,2].flat, data[np_idx,-1].flat,'rX')

    ax.set_xlabel(r'$x_{1}$' + 'data')
    ax.set_ylabel(r'$x_{2}$' + 'data')
    ax.set_zlabel('y')

class Affine:
    def __init__(self):
        self._feature_dim = feature_dim

        self._z1_list = [None]*(self._feature_dim+1)
        self._z2_list = self._z1_list.copy()

        self._dz1_list, self._dz2_list = self._z1_list.copy(), self._z1_list.copy()
        self._dth_list = self._z1_list.copy()

        self._node_imp()
        self._random_initialization()

    def _node_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def _random_initialization(self):
        r_feature_dim = 1/np.power(self._feature_dim, 0.5)
        self._Th = np.random.uniform(low = -1*r_feature_dim,
                                     high = r_feature_dim,
                                     size = (self._feature_dim+1, 1))

    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:,node_idx])

        self._z2_list[1] = self._node2[1].forward(self._Th[0], self._z1_list[1])

        for node_idx in range(2, self._feature_dim + 1):
            self._z2_list[node_idx] = self._node2[node_idx].forward(self._z2_list[node_idx-1], self._z1_list[node_idx])

        return self._z2_list[-1]

    def backward(self, dz2_last, lr):
        self._dz2_list[-1] = dz2_last
        for node_idx in reversed(range(1, self._feature_dim+1)):
            dz2, dz1 = self._node2[node_idx].backward(self._dz2_list[node_idx])
            self._dz2_list[node_idx-1] = dz2
            self._dz1_list[node_idx] = dz1

        self._dth_list[0] = self._dz2_list[0]
        for node_idx in reversed(range(1, self._feature_dim+1)):
            dth, _ = self._node1[node_idx].backward(self._dz1_list[node_idx])
            self._dth_list[node_idx] = dth

        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] = self._Th[th_idx] - lr * np.sum(self._dth_list[th_idx])

        return self._Th

    def get_Th(self):
        return self._Th

class Sigmoid:
    def __init__(self):
        self._pred = None

    def forward(self, z):
        self._pred = 1/(1 + np.exp(-1*z))
        return self._pred

    def backward(self, dpred):
        partial = self._pred * (1-self._pred)
        dz = dpred * partial
        return dz

class MVLoR:
    def __init__(self):
        self._feature_dim = feature_dim
        self._affine = Affine()
        self._sigmoid = Sigmoid()

    def forward(self, x):
        z = self._affine.forward(x)
        pred = self._sigmoid.forward(z)
        return pred

    def backward(self, dpred, lr):
        dz = self._sigmoid.backward(dpred)
        self._affine.backward(dz, lr)

    def get_Th(self):
        return self._affine.get_Th()

class BinaryCrossEntropy_Loss:
    def __init__(self):
        self._y, self._pred = None, None
        self._mean_node = nodes.mean_node()

    def forward(self, y, pred):
        self._y, self._pred = y, pred
        loss = -1*(self._y*np.log(self._pred) + (1-self._y)*np.log(1-self._pred))
        J = self._mean_node.forward(loss)
        return J

    def backward(self):
        dloss = self._mean_node.backward(1)
        dpred = dloss*(self._pred - self._y)/(self._pred*(1-self._pred))
        return dpred

def result_tracker():
    global iter_idx, check_freq
    global th_accum, model
    if iter_idx % check_freq == 0:
        th_accum = np.hstack((th_accum, model.get_Th()))
        cost_list.append(J)
    iter_idx += 1

def plot_classifier():
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')

    ax.plot(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 'bo')
    ax.plot(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 'rX')
    ax.set_xlabel(r'$x_{1}$' + ' data', labelpad=10)
    ax.set_ylabel(r'$x_{2}$' + ' data', labelpad=10)
    ax.set_zlabel('y', labelpad=10)

    f_th0, f_th1, f_th2 = th_accum[:, -1]
    x1_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    x2_range = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    affine = X2 * f_th2 + X1 * f_th1 + f_th0
    pred = Sigmoid.forward(affine)
    ax.plot_wireframe(X1, X2, pred)

def result_visualizer():
    global th_accum, loss_list, cost_list

    fig, ax = plt.subplots(figsize = (30,10))
    fig.subplots_adjust(hspace = 0.3)
    ax.set_title(r'$\vec{\theta}$' + ' Update')

    for feature_idx in range(feature_dim+1):
        ax.plot(th_accum[feature_idx,:], label = r'$\theta_{%d}$'%feature_idx)
    ax.legend()
    iter_ticks = np.linspace(0,th_accum.shape[1],10).astype(np.int)
    ax.set_xticks(iter_ticks)

feature_dim = 5
noise_factor = 0.5
direction = 1
n_sample = 500

#x_dict = {1:{'mean':0, 'std':2},2:{'mean':0, 'std':2}}
#t_th_list = [0,1,2]

data_gen = dataset_generator(feature_dim = feature_dim, n_sample = n_sample, noise_factor = noise_factor, direction = direction)
data = data_gen.make_dataset()
#data_gen.set_t_th(t_th_list)
#data_gen.set_feature_dict(x_dict)
#dataset_visualizer()
batch_size = 8
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)

model = MVLoR()
BCE_loss = BinaryCrossEntropy_Loss()

th_accum = model.get_Th()
cost_list = []

epochs, lr = 200, 0.05
iter_idx, check_freq = 0, 10

for epochs in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)

        X, Y = batch[:,:-1], batch[:,-1]

        # Forward propagation
        pred = model.forward(X)
        J = BCE_loss.forward(Y, pred)

        # Backward propagation
        dpred = BCE_loss.backward()
        model.backward(dpred, lr)

        # Result Tracking
        result_tracker()

result_visualizer()
