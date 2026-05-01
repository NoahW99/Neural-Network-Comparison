import numpy as np
from scipy.special import expit

class LLMNeuralNetwork:
    def __init__(self, dims, activation='sigmoid', lr=0.01, class_weights=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.dims = dims
        self.lr = lr
        self.class_weights = class_weights
        self.params = {}
        for i in range(1, len(dims)):
            # He init for relu, Xavier-like otherwise
            scale = np.sqrt(2.0 / dims[i-1]) if activation == 'relu' else np.sqrt(1.0 / dims[i-1])
            self.params[f'W{i}'] = np.random.randn(dims[i-1], dims[i]) * scale
            self.params[f'b{i}'] = np.zeros((1, dims[i]))
        self.activation = activation

    def parameters(self):
        """
        Yield every learnable ndarray (weights and biases) so that
        utilities like count_learnable_params() can iterate over them.
        """
        # Get the number of layers from params dictionary
        n_layers = len([k for k in self.params.keys() if k.startswith('W')])

        # Yield weights and biases in order
        for i in range(1, n_layers + 1):
            yield self.params[f'W{i}']
            yield self.params[f'b{i}']

    def _activate(self, Z):
        if self.activation == 'sigmoid':
            return expit(Z)
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