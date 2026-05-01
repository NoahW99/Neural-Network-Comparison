import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchModel(nn.Module):
    """
    Simple feedforward neural network with two hidden layers.
    """
    def __init__(self, input_features=4, h1=8, h2=9, output_features=3, class_weights=None):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

        # weights for classes
        if class_weights is not None:
            weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
            weight_tensor = torch.FloatTensor(weight_list)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass with ReLU activations.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


    def compute_loss(self, outputs, labels):
        """
        Compute (weighted) cross-entropy loss.
        """
        return self.criterion(outputs, labels)


    def backward(self, loss, optimizer):
        """
        Performs a backward propagation step:
        - Clears previous gradients
        - Computes gradients via autograd
        - Updates parameters with the given optimizer
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()