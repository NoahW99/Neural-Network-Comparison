import torch
import torch.nn as nn
import torch.nn.functional as F

# Create Model Class that inherits from nn.Module

class TorchModel(nn.Module):
    #Input layer --> hidden layer (number of neurons)

    def __init__(self, input_features=4, h1=8, h2=9, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))    # rectified linear unit
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x