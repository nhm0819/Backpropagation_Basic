import numpy as np

score_list = np.random.randint(0, 10, size=(1000))
print(score_list)

M, m = None, None
for score in score_list:
    if M==None or score > M:
        M = score
    if m==None or score < m:
        m = score
print(M, m)

cnt_list = [0 for i in range(M-m)]
print(cnt_list, len(cnt_list))
for score in score_list:
    cnt_list[score-1] = cnt_list[score-1] + 1
print(cnt_list)



cnt_dict = dict()

for score in score_list:
    cnt_dict[score] = cnt_dict.get(score, 0) + 1
print(cnt_dict)

key_list = list(cnt_dict.keys())
val_list = list(cnt_dict.values())
item_list = list(cnt_dict.items())

print(key_list[0])
M = None
M_score = None
for k, v in cnt_dict.items():
    print(k, v)
    if M_score is None or k > M_score:
        M_score = k
        M = v
print('Max Score is', M_score,'and count', M)

## initializer
class test_class:
    def __init__(self, x, y):
        print(x, y)

tmp1 = test_class(10, 20)


## class variable
class node:
    node_cnt = 0

    def __init__(self, x, y):
        self.x , self.y = x, y
        self.add = None

        self.adder()

        node.node_cnt += 1

    def adder(self):
        self.add = self.x + self.y

node1 = node(10, 20)
print(node1.x, node1.y, node1.node_cnt)
node2 = node(100, 200)
print(node2.x, node2.y, node2.node_cnt)
node3 = node(200, 300)
print(node3.x, node3.y, node3.node_cnt, node3.add)


## underbar is private mark.
class calculator:
    def __init__(self, x, y, operator):
        self._x = x
        self._y = y
        self.operator = operator

        self.result = None

        if operator == '+':
            self._adder()
        elif operator == '-':
            self._subtractor()
        elif operator == '*':
            self._multiplier()

    def _adder(self):
        self.result = self._x + self._y

    def _subtractor(self):
        self.result = self._x - self._y

    def _multiplier(self):
        self.result = self.x * self.y

    def get_result(self):
        return self.result

tmp = calculator(10, 20, '+')
print(tmp.get_result())

class tmp_class:
    def __init__(self):
        self.x == 0
        self.y == 0
    def method1():
        pass
    def method2():
        pass

tmp = tmp_class
print(dir(tmp))

a=10+20j
print(dir(a))

print(a.real)
print(a.imag)



score_list = [50, 70, 40, 90, 95, 20, 15, 65, 77]
class grader:
    def __init__(self, score_list):
        self.score_list = score_list

        self._grade_list = None
        self._cutoff_list = None

        self._init_grader()

    def _init_grader(self):
        self._grade_list = ['A', 'B', 'C', 'F']
        self._cutoff_list = [100, 90, 80, 70, 0]

    def score2grade(self):
        return_grade_list = []
        for score_idx, score in enumerate(self.score_list):
            if score >= self._cutoff_list[1] and \
                score <= self._cutoff_list[0]:
                return_grade_list.append(self._grade_list[0])
            elif score >= self._cutoff_list[2] and \
                score <= self._cutoff_list[1]:
                return_grade_list.append(self._grade_list[1])
            elif score >= self._cutoff_list[3] and \
                score <= self._cutoff_list[2]:
                return_grade_list.append(self._grade_list[2])
            elif score >= self._cutoff_list[4] and \
                score <= self._cutoff_list[3]:
                return_grade_list.append(self._grade_list[3])
            else:
                class scoreError(Exception):
                    pass
                raise scoreError("Invalid Score Detected!")
                #return_grade_list.append("NR")

        print(return_grade_list)

tmp= grader(score_list)
tmp.score2grade()

import numpy as np
import matplotlib.pyplot as plt

N = 100
n_feature = 3
x_data = np.random.normal(0, 1, size=(N,n_feature))
y_data = np.sum(x_data, axis=1) + 0.2*np.random.normal(0,1,size=(N,1))
print(np.sum(x_data, axis=1))
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(x_data, y_data, 'bo')
plt.show()

data = np.hstack((x_data, y_data))
print(data.shape)
print(x_data.shape, y_data.shape)

x_data = data[:,0]
y_data = data[:,1]


class dataset_generator:
    def __init__(self, feature_dim = 1, n_sample=100, noise=0):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise = noise

        self._coefficient = None
        self.__init_set_coefficient()

    def __init_set_coefficient(self):
        self._coefficient = [1 for _ in range(self._feature_dim)] + [0]


    def set_n_sample(self, n_sample):
        self._n_sample = n_sample

    def set_noise(self, noise):
        self._noise = noise

    def set_coefficient(self, coefficient_list):
        self._coefficient = coefficient_list

    def make_dataset(self):
        x_data = np.random.normal(0, 1, size=(self._n_sample, self._feature_dim))

        y_data = np.zeros(shape=(self._n_sample, 1))
        for feature_idx in range(self._feature_dim):
            y_data += self._coefficient[feature_idx]*x_data[:,feature_idx].reshape(-1,1)
        y_data += self._coefficient[-1]
        y_data += self._noise*np.random.normal(0,1, size=(self._n_sample, 1))

        return x_data, y_data

data_gen = dataset_generator(feature_dim=3)
x_data, y_data = data_gen.make_dataset()

print(data_gen._coefficient)

data_gen.set_coefficient([2,-1,3,5])
print(data_gen._coefficient)

print(x_data.shape, y_data.shape)

plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(x_data, y_data, 'bo', alpha=0.3, markersize=20)
plt.show()

