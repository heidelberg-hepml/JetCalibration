import torch
import torch.nn as nn


def MLP(input_dim, output_dim, hidden_dim, num_layers, drop=0.):
    """
    Creates a multi-layer perceptron (MLP) neural network.

    Constructs a feedforward neural network with a configurable number of hidden layers.
    Each hidden layer has the same dimension and uses ReLU activation.
    The final layer projects to the specified output dimension with no activation.

    Args:
        input_dim (int): Dimension of the input features
        output_dim (int): Dimension of the network output 
        hidden_dim (int): Number of neurons in each hidden layer
        num_layers (int): Number of hidden layers in the network

    Returns:
        nn.Sequential: PyTorch Sequential model containing the MLP layers

    Architecture:
        - Input layer: input_dim neurons
        - num_layers hidden layers with:
            - Linear transformation to hidden_dim neurons
            - ReLU activation
        - Output layer: Linear transformation to output_dim neurons
    """
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if drop != 0.:
            layers.append(nn.Dropout(drop))
        input_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


