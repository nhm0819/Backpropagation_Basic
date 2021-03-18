# import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from LR_dataset_generator import LR_dataset_generator as dataset_generator
import basic_nodes as nodes


plt.style.use('seaborn')

# dataset preperation
np.random.seed(0)

def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx*batch_size:]
    else:
        batch = data[batch_idx*batch_size:(batch_idx+1)*batch_size]
    return batch

feature_dim = 5
batch_size = 8
batch_idx = 0
dataset_gen = dataset_generator(feature_dim=feature_dim)

data = dataset_gen.make_dataset()
x_data, y_data = data[:,:feature_dim+1], data[:,-1]

n_batch = np.ceil(data.shape[0]/batch_size).astype(int)

# model implementation
node1 = [None] + [nodes.mul_node() for _ in range(feature_dim)]
node2 = [None] + [nodes.plus_node() for _ in range(feature_dim)]

# cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()


class Affine_Function:
    def __init__(self, feature_dim, Th):
        self._feature_dim = feature_dim
        self._Th = Th

        self._Z1_list = [None]*(self._feature_dim + 1)
        self._Z2_list = self._Z1_list.copy()
        self._dZ1_list, self._dZ2_list = self._Z1_list.copy(), self._Z1_list.copy()
        self._dTh_list = self._dZ1_list.copy()

        self.affine_imp()

    def affine_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._Z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:,node_idx])

        self._Z2_list[1] = self._node2[1].forward(self._Th[0], self._Z1_list[1])

        for node_idx in range(2, self._feature_dim + 1):
            self._Z2_list[node_idx] = self._node2[node_idx].forward(self._Z2_list[node_idx-1], self._Z1_list[node_idx])

        return self._Z2_list[-1]

    def backward(self, dZ2_last, lr):
        self._dZ2_list[-1] = dZ2_last
        for node_idx in reversed(range(1, self._feature_dim+1)):
            dZ2, dZ1 = self._node2[node_idx].backward(self._dZ2_list[node_idx])
            self._dZ2_list[node_idx-1] = dZ2
            self._dZ1_list[node_idx] = dZ1

        self._dTh_list[0] = self._dZ2_list[0]
        for node_idx in reversed(range(1, self._feature_dim+1)):
            dTh, _ = self._node1[node_idx].backward(self._dZ1_list[node_idx])
            self._dTh_list[node_idx] = dTh

        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] = self._Th[th_idx] - lr*np.sum(self._dTh_list[th_idx])

        return self._Th

class MSE_Cost:
    def __init__(self):
        self.cost_imp()

    def cost_imp(self):
        self._node3 = nodes.minus_node()
        self._node4 = nodes.square_node()
        self._node5 = nodes.mean_node()

    def forward(self, Y, pred):
        Z3 = self._node3.forward(Y, pred)
        L = self._node4.forward(Z3)
        J = self._node5.forward(L)
        return J

    def backward(self):
        dL = self._node5.backward(1)
        dZ3 = self._node4.backward(dL)
        _, dZ2_last = self._node3.backward(dZ3)
        return dZ2_last


# hyperparameter settings
Th = np.random.normal(0,1,size = (feature_dim+1)).reshape(-1,1)

epochs = 100
lr = 0.01

loss_list = []
cost_list = []
th_accum = np.array(Th).reshape(-1, 1)

affine = Affine_Function(feature_dim, Th)
cost = MSE_Cost()

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)
        X, Y = batch[:, :-1], batch[:, -1]

        Pred = affine.forward(X)
        J = cost.forward(Y, Pred)

        dPred = cost.backward()
        affine.backward(dPred, lr)

        th_accum = np.hstack((th_accum, affine._Th))
        cost_list.append(J)



def result_visualization(th_accum, loss_list):
    fig, ax = plt.subplots(2, 1, figsize = (30,15))
    for i in range(feature_dim+1):
        ax[0].plot(th_accum[i], label = r'$\theta_{%d}$'%i, linewidth = 1)
    ax[1].plot(cost_list)
    ax[0].legend(loc='lower right', fontsize=10)
    ax[0].tick_params(axis='both', labelsize = 10)
    ax[1].tick_params(axis='both', labelsize = 10)
    ax[0].set_title(r'$\vec{\theta}$', fontsize=20)
    ax[1].set_title('Cost', fontsize=20)

result_visualization(th_accum, cost_list)