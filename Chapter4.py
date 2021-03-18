# import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset_generator import dataset_generator
import basic_nodes as nodes

# hyperparameter settings
t_th1, t_th0 = 5, 5
th1, th0 = 1, 1
epochs = 20
lr = 0.01
batch_size = 8
check_freq = 3

loss_list = []
cost_list = []
th0_list, th1_list = [], []


# dataset preperation
np.random.seed(0)

dataset_gen = dataset_generator()

dataset_gen.set_coefficient([t_th1,t_th0])

x_data, y_data = dataset_gen.make_dataset()
data = np.hstack((x_data, y_data))
n_batch = np.ceil(data.shape[0]/batch_size).astype(int)

plt.style.use('seaborn')

# model implementation
node1 = nodes.mul_node()
node2 = nodes.plus_node()

# square loss / MSE cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()

# Mini-batch Extraction
for epoch in range(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        batch = data[batch_idx*batch_size : (batch_idx+1)*batch_size]

        z1 = node1.forward(th1, batch[:,0])
        z2 = node2.forward(z1, th0)
        z3 = node3.forward(batch[:,1], z2)
        L = node4.forward(z3)
        J = node5.forward(L)

        dL = node5.backward(1)
        dz3 = node4.backward(dL)
        dy, dz2 = node3.backward(dz3)
        dz1, dth0 = node2.backward(dz2)
        dth1, dx = node1.backward(dz1)

        th1 = th1 - lr*np.sum(dth1)
        th0 = th0 - lr*np.sum(dth0)

        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(L)
        cost_list.append(J)


fig, ax = plt.subplots(2, 1, figsize = (20,15))
ax[0].plot(th1_list, label = r'$\theta_{1}$')
ax[0].plot(th0_list, label = r'$\theta_{0}$')
ax[1].plot(cost_list)

title_font = {'size':30, 'alpha':0.8, 'color':'navy'}
label_font = {'size':15, 'alpha':0.8, 'color':'red'}

ax[0].set_title(r'$\theta_{1}$, $\theta_{0}$', fontdict = title_font)
ax[1].set_title("loss", fontdict = title_font)
ax[1].set_xlabel("epoch", fontdict = label_font)



N_line = 200
cmap = cm.get_cmap('rainbow', lut = N_line)

fig, ax = plt.subplots(1, 1, figsize = (10,10))
ax.scatter(x_data, y_data)

test_th = th_list[:N_line]
x_range = np.array([np.min(x_data), np.max(x_data)])

for line_idx in range(N_line):
    pred_line = np.array([x_range[0]*test_th[line_idx],
                          x_range[1]*test_th[line_idx]])
    ax.plot(x_range, pred_line, color = cmap(line_idx), alpha = 0.1)



def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx*batch_size : ]
    else:
        batch = data[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch

# SVLR class
class SVLR:
    def __init__(self, th1, th0):
        self.th1, self.th0 = th1, th0

        self.th1_list, self.th0_list = [], []
        self.cost_list = []

        self.iter_cnt, self.check_cnt = 0, 0

        self.model_imp()
        self.cost_imp()

    def model_imp(self):
        self.node1 = nodes.mul_node()
        self.node2 = nodes.plus_node()

    def cost_imp(self):
        self.node3 = nodes.minus_node()
        self.node4 = nodes.square_node()
        self.node5 = nodes.mean_node()

    def forward(self, mini_batch):
        Z1 = self.node1.forward(self.th1, mini_batch[:,0])
        Z2 = self.node2.forward(Z1, self.th0)
        Z3 = self.node3.forward(mini_batch[:,1], Z2)
        L = self.node4.forward(Z3)
        J = self.node5.forward(L)

        if self.iter_cnt % check_freq == 0:
            self.cost_list.append(J)

    def backward(self, lr):
        if self.iter_cnt % check_freq == 0:
            self.th1_list.append(self.th1)
            self.th0_list.append(self.th0)
            self.check_cnt += 1

        dL = self.node5.backward(1)
        dZ3 = self.node4.backward(dL)
        dY , dZ2 = self.node3.backward(dZ3)
        dZ1, dTh0 = self.node2.backward(dZ2)
        dTh1, dX = self.node2.backward(dZ1)

        self.th1 = self.th1 - lr*np.sum(dTh1)
        self.th0 = self.th0 - lr*np.sum(dTh0)

        self.iter_cnt += 1

    def result_visualization(self):
        fig, ax = plt.subplots(2, 1, figsize = (30,15))
        ax[0].plot(self.th1_list, label = r'$\theta_{1}$')
        ax[0].plot(self.th0_list, label = r'$\theta_{0}$')
        ax[1].plot(self.cost_list)
        ax[1].legend(loc = "lower right", fontsize = 30)

def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx*batch_size : ]
    else:
        batch = data[batch_idx*batch_size : (batch_idx+1)*batch_size]
    return batch

model = SVLR(th1, th0)

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)

        model.forward(batch)
        model.backward(lr)

model.result_visualization()