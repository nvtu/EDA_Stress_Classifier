import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


"""
Definition of AutoEncoder Network structure
"""

class Decoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimensions, code_dimension):
        """
        Initialize decoder's parameters
            - input_dimension: Integer - The input layer (should be equal to the code_dimension of the Encoder Network)
            - hidden_dimension: A list of integers that define the number of nodes in each hidden layer
            - code_dimension: Integer - The output layer (should be equal to the input_dimension of the Encoder Network)
        """
        super(Decoder, self).__init__()
        D = []
        network_dimensions = [input_dimension, *hidden_dimensions, code_dimension] # We will iterate this list reversely
        num_dimension = len(network_dimensions)
        for layer_index in range(num_dimension - 1):
            next_layer_index = layer_index + 1
            # layer -> activation function -> batch norm layer
            decoder_layer = nn.Linear(network_dimensions[num_dimension - 1 - layer_index], network_dimensions[num_dimension - 1 - next_layer_index]) # Reverse indices
            activation_function = nn.ELU()
            decoder_batch_norm1d = nn.BatchNorm1d(network_dimensions[num_dimension - 1 - next_layer_index])
            DL = [decoder_layer, activation_function, decoder_batch_norm1d] if layer_index + 1 < num_dimension - 1 else [decoder_layer] # We do not add activation function or batch norm layer at the end of the network
            D += DL

        self.decoder = nn.Sequential(*D)


    def forward(self, X):
        out = self.decoder(X)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimensions, code_dimension):
        """
        Initialize decoder's parameters
            - input_dimension: Integer - The input layer
            - hidden_dimension: A list of integers that define the number of nodes in each hidden layer
            - code_dimension: Integer - The output layer
        """
        super(Encoder, self).__init__()
        E = []
        network_dimensions = [input_dimension, *hidden_dimensions, code_dimension]
        num_dimension = len(network_dimensions)
        for layer_index in range(num_dimension - 1):
            next_layer_index = layer_index + 1
            # layer -> activation function -> batch norm layer
            encoder_layer = nn.Linear(network_dimensions[layer_index], network_dimensions[next_layer_index])
            activation_function = nn.ELU()
            encoder_batch_norm1d = nn.BatchNorm1d(network_dimensions[next_layer_index]) 
            decoder_batch_norm1d = nn.BatchNorm1d(network_dimensions[num_dimension - 1 - next_layer_index])
            EL = [encoder_layer, activation_function, encoder_batch_norm1d] if layer_index + 1 < num_dimension - 1 else [encoder_layer] # We do not add activation function or batch norm layer at the end of the network
            E += EL
        self.encoder = nn.Sequential(*E)


    def forward(self, X):
        out = self.encoder(X)
        return out