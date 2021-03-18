# import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import basic_nodes as nodes


plt.style.use('seaborn')


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


def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx*batch_size:]
    else:
        batch = data[batch_idx*batch_size:(batch_idx+1)*batch_size]
    return batch

# dataset preperation
np.random.seed(0)

n_sample = 200

x_data1 = np.linspace(0.05, 1-0.05, n_sample).reshape(-1,1)
y_data = np.sin(2*np.pi*x_data1) + 0.2*np.random.normal(0, 1, size=(n_sample,1))

x_data = np.zeros(shape = (n_sample,1))
h_order = 5
for order in range(1, h_order+1):
    order_data = np.power(x_data1, order)
    x_data = np.hstack((x_data, order_data))

data = np.hstack((x_data, y_data))

feature_dim = x_data.shape[1]-1
batch_size = 32

n_batch = np.ceil(data.shape[0]/batch_size).astype(int)


# hyperparameter settings
Th = np.ones(shape=(feature_dim +1,), dtype=np.float).reshape(-1,1)
epochs = 100
lr = 0.01
th_accum = Th.reshape(-1, 1)
loss_list = []
cost_list = []


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
    fig, ax = plt.subplots(2, 1, figsize = (40,20))
    for i in range(feature_dim+1):
        ax[0].plot(th_accum[i], label = r'$\theta_{%d}$'%i)
    ax[1].plot(cost_list)
    ax[0].legend(loc='lower right', fontsize=10)
    ax[0].tick_params(axis='both', labelsize = 10)
    ax[1].tick_params(axis='both', labelsize = 10)
    ax[0].set_title(r'$\vec{\theta}$', fontsize=20)
    ax[1].set_title('Cost', fontsize=20)

result_visualization(th_accum, cost_list)


fig, ax = plt.subplots(figsize = (20,20))
ax.plot(x_data1, y_data, 'bo')

cmap = cm.get_cmap('rainbow', lut = th_accum.shape[1])
x_range = np.linspace(np.min(x_data1), np.max(x_data1),100)

for th_idx in range(0, th_accum.shape[1], 10):
    pred = np.zeros(shape = x_range.shape)
    for i in range(th_accum.shape[0]):
        th = th_accum[i,th_idx]
        pred += th*np.power(x_range,i)

    ax.plot(x_range, pred, color = cmap(th_idx), alpha = 0.1)

ax.tick_params(axis = 'both', labelsize = 30)