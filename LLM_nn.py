import numpy as np

class LLMNeuralNetwork:
    def __init__(self, dims, activation='sigmoid', lr=0.01, class_weights=None):
        self.dims = dims
        self.lr = lr
        self.class_weights = class_weights
        self.params = {}
        for i in range(1, len(dims)):
            self.params[f'W{i}'] = np.random.randn(dims[i-1], dims[i]) * 0.01
            self.params[f'b{i}'] = np.zeros((1, dims[i]))
        self.activation = activation

    def _activate(self, Z):
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(-Z))
        return np.maximum(0, Z)

    def forward(self, X):
        A = X
        cache = {'A0': X}

        for i in range(1, len(self.dims)-1):
            Zi = A.dot(self.params[f'W{i}']) + self.params[f'b{i}']
            Ai = self._activate(Zi) 
            cache[f'Z{i}'] = Zi
            cache[f'A{i}'] = Ai
            A = Ai

        i = len(self.dims) - 1 
        ZL = A.dot(self.params[f'W{i}']) + self.params[f'b{i}']
        exp = np.exp(ZL - np.max(ZL, axis=1, keepdims=True))
        AL = exp / exp.sum(axis=1, keepdims=True)
        cache['ZL'] = ZL
        cache['AL'] = AL

        return AL, cache

    def compute_loss(self, AL, y):
        m = y.shape[0]
        log_likelihood = -np.log(AL[range(m), y])

        if self.class_weights is None:
            return np.sum(log_likelihood)/m
        sw = np.vectorize(self.class_weights.get)(y)

        return np.sum(sw * log_likelihood) / m

    def backward(self, X, y, cache):
        grads = {}
        m = X.shape[0]
        Y = np.zeros_like(cache['AL']); Y[np.arange(m), y] = 1
        dZ = cache['AL'] - Y

        if self.class_weights is not None:
            sw = np.vectorize(self.class_weights.get)(y)         
            dZ = dZ * sw[:, np.newaxis] 

        for i in reversed(range(1, len(self.dims))):
            grads[f'dW{i}'] = (cache[f'A{i-1}'].T.dot(dZ))/m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)/m
            dA_prev = dZ.dot(self.params[f'W{i}'].T)
            dZ = dA_prev * (cache[f'A{i-1}'] * (1-cache[f'A{i-1}'])) if self.activation=='sigmoid' else (dA_prev * (cache[f'A{i-1}']>0))
        return grads

    def update(self, grads):
        for i in range(1, len(self.dims)):
            self.params[f'W{i}'] -= self.lr * grads[f'dW{i}']
            self.params[f'b{i}']   -= self.lr * grads[f'db{i}']

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            AL, cache = self.forward(X)
            grads = self.backward(X, y, cache)
            self.update(grads)

    def predict(self, X):
        AL, _ = self.forward(X)
        return np.argmax(AL, axis=1)