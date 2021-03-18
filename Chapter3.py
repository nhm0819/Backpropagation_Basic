# import required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset_generator import dataset_generator
import basic_nodes as nodes


# dataset preperation
dataset_gen = dataset_generator()

dataset_gen.set_coefficient([5,0])

x_data, y_data = dataset_gen.make_dataset()

# plt.style.use('seaborn')
# fig, ax = plt.subplots(figsize = (10,10))
# ax.plot(x_data, y_data, 'bo')
# plt.show()

# model implementation
node1 = nodes.mul_node()

# square error loss implementation
node2 = nodes.minus_node()
node3 = nodes.square_node()

node4 = nodes.mean_node()

# hyperparameter settings
epochs = 200
iterations = 200

batch_size = 16
n_batch = int(np.ceil(len(x_data)/batch_size))
t_iteration = 500
epochs = np.ceil(t_iteration/n_batch).astype(int)

lr = 0.01
th = -1
loss_list = []
cost_list = []
th_list = []

# Mini-Batch Gradient Descent
for epoch in range(epochs):
    idx_arr = np.arange(len(x_data))
    np.random.shuffle(idx_arr)
    ## with Replacement
    # random_idx = np.random.choice(idx_arr, batch_size)
    x_data = x_data[idx_arr]
    y_data = y_data[idx_arr]
    for batch_idx in range(n_batch):
        if batch_size is n_batch-1:
            X = x_data[batch_idx*batch_size : ]
            Y = y_data[batch_idx*batch_size : ]
        else:
            X = x_data[batch_idx*batch_size : (batch_idx+1)*batch_size]
            Y = y_data[batch_idx*batch_size : (batch_idx+1)*batch_size]

        Z1 = node1.forward(th, X)
        Z2 = node2.forward(Y, Z1)
        L = node3.forward(Z2)
        J = node4.forward(L)

        dL = node4.backward(1)
        dZ2 = node3.backward(dL)
        dY, dZ1 = node2.backward(dZ2)
        dTh, dX = node1.backward(dZ1)

        th = th - lr*np.sum(dTh)

        th_list.append(th)
        cost_list.append(J)

# # Stochastic Gradient Descent with Replacement
# for _ in range(iterations):
#     idx_arr = np.arange(len(x_data))
#     random_idx = np.random.choice(idx_arr, 1)
#     np.random.shuffle(random_idx)
#     x_data = x_data[random_idx]
#     y_data = y_data[random_idx]
#
#     Z1 = node1.forward(th, X)
#     Z2 = node2.forward(Y, Z1)
#     L = node3.forward(Z2)
#
#     dZ2 = node3.backward(1)
#     dY, dZ1 = node2.backward(dZ2)
#     dTh, dX = node1.backward(dZ1)
#
#     th = th - lr*th
#
#     th_list.append(th)
#     loss_list.append(l)
#
# # Batch Gradient Descent
# for _ in range(epochs):
#     X, Y = x_data, y_data
#
#     Z1 = node1.forward(th, X)
#     Z2 = node2.forward(Y, Z1)
#     L = node3.forward(Z2)
#     J = node4.forward(L)
#
#     dL = node4.backward(1)
#     dZ2 = node3.backward(dL)
#     dY, dZ1 = node2.backward(dZ2)
#     dTh, dX = node1.backward(dZ1)
#
#     th = th - lr*np.sum(dTh)
#
#     th_list.append(th)
#     cost_list.append(J)

plt.style.use('seaborn')
fig, ax = plt.subplots(2, 1, figsize = (30, 10))
ax[0].plot(th_list, linewidth = 5)
ax[1].plot(cost_list, linewidth = 5)

title_font = {'size':30, 'alpha':0.8, 'color':'navy'}
label_font = {'size':15, 'alpha':0.8, 'color':'red'}


ax[0].set_title(r'$\theta$', fontdict = title_font)
ax[1].set_title("Cost", fontdict = title_font)
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


