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
