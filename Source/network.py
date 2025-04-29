import torch
import torch.nn as nn


def MLP(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


