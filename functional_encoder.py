# By Jackson Thissell
# 12/11/2023

import torch.nn as nn
import torch.nn.functional as F


# This is a simple fully-connected auto-encoder
# Meant for testing the convergence of fractal dimension.
class FunctionalAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size, layer_num, layer_size):
        super(FunctionalAutoEncoder, self).__init__()

        self.en_fc1 = nn.Linear(input_size, layer_size)

        self.en_layers = nn.ModuleList([])
        for i in range(layer_num):
            self.en_layers.append(nn.Linear(layer_size, layer_size))

        self.en_fc2 = nn.Linear(layer_size, latent_size)

        self.de_fc1 = nn.Linear(latent_size, layer_size)

        self.de_layers = nn.ModuleList([])
        for i in range(layer_num):
            self.de_layers.append(nn.Linear(layer_size, layer_size))

        self.de_fc2 = nn.Linear(layer_size, input_size)

    def encode(self, x):
        x = F.relu(self.en_fc1(x))

        for en_layer in self.en_layers:
            x = F.relu(en_layer(x))

        return self.en_fc2(x)

    def decode(self, x):
        x = F.relu(self.de_fc1(x))

        for de_layer in self.de_layers:
            x = F.relu(de_layer(x))

        return self.de_fc2(x)

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)
