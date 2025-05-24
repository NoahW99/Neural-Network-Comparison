import numpy as np

class NeuralNetwork:
    """
    A simple feedforward neural network implemented from scratch.
    Arguments:
      layer_sizes: list of ints defining the network architecture, e.g. [input_dim, hidden1, ..., output_dim]
      activations: list of strings of length L (number of layers), choices: 'sigmoid', 'relu'
      lr: learning rate
      epochs: number of training epochs

    Inspiration taken from https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc
    """
    def __init__(self, layer_sizes, activations, lr=0.01, epochs=1000, class_weights=None):
        assert len(layer_sizes) >= 2, "Need at least input and output layer"
        assert len(activations) == len(layer_sizes) - 1, \
            "Must specify one activation per layer transition"

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr
        self.epochs = epochs
        self.class_weights = class_weights

        # initialize weights and biases
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W.append(np.random.uniform(-limit, limit, (out_dim, in_dim)))
            self.b.append(np.zeros((out_dim, 1)))

    def _activate(self, Z, func):
        if func == 'relu':
            A = np.maximum(0, Z)
            dA = (Z > 0).astype(float)
        elif func == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            dA = A * (1 - A)
        else:
            raise ValueError(f"Unsupported activation: {func}")
        return A, dA

    def feed_forward(self, X):
        """
        X: numpy array, shape (n_features, m)
        Returns output A_L and cache of intermediates.
        """
        A = X
        cache = {'A0': A}
        for i, func in enumerate(self.activations):
            Z = self.W[i] @ A + self.b[i]
            A, dA = self._activate(Z, func)
            cache[f'Z{i+1}'] = Z
            cache[f'A{i+1}'] = A
            cache[f'dA{i+1}'] = dA
        return A, cache

    def compute_cost(self, A_L, Y):
        """Binary cross-entropy cost for one output or multi-output average"""
        m = Y.shape[1]
        # avoid log(0)
        eps = 1e-8
        loss_matrix = -(Y * np.log(A_L + eps) + (1 - Y) * np.log(1 - A_L + eps))
        cost = np.sum(loss_matrix) / m
        if self.class_weights is None:
            cost = np.sum(loss_matrix) / m
        else: 
            # get weights per sample
            true_labels    = np.argmax(Y, axis=0)
            sample_weights = np.vectorize(self.class_weights.get)(true_labels)
            per_sample_loss = np.sum(loss_matrix, axis=0)
            cost = np.sum(sample_weights * per_sample_loss) / m
        return cost

    def backprop(self, cache, Y):
        """
        Compute gradients for all layers.
        Returns lists dW and db matching self.W and self.b
        """
        L = len(self.W)
        m = Y.shape[1]
        grads_W = [None] * L
        grads_b = [None] * L

        # initialize dA from output layer
        A_L = cache[f'A{L}']
        dZ = (A_L - Y) / m
        
        if self.class_weights is not None:
            true_labels = np.argmax(Y, axis=0)
            sw = np.vectorize(self.class_weights.get)(true_labels)
            dZ = dZ * sw[np.newaxis, :] 

        # backprop through layers L..1
        for i in reversed(range(L)):
            A_prev = cache[f'A{i}']
            grads_W[i] = dZ @ A_prev.T
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True)
            if i > 0:
                dA_prev = self.W[i].T @ dZ
                dZ = dA_prev * cache[f'dA{i}']
        return grads_W, grads_b

    def update_parameters(self, grads_W, grads_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def train(self, X, Y, print_every=100):
        """
        Train the network with input X (n_x, m) and labels Y (n_y, m).
        """
        for epoch in range(1, self.epochs + 1):
            A_L, cache = self.feed_forward(X)
            cost = self.compute_cost(A_L, Y)
            grads_W, grads_b = self.backprop(cache, Y)
            self.update_parameters(grads_W, grads_b)
            if print_every and epoch % print_every == 0:
                print(f"Epoch {epoch}/{self.epochs} - Cost: {cost:.6f}")
        return self

    def predict(self, X, threshold=0.5):
        """Return predictions (0/1) or probabilities if threshold=None"""
        A_L, _ = self.feed_forward(X)
        if threshold is None:
            return A_L
        return (A_L >= threshold).astype(int)