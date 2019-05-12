# +-----------------------------------------------------------------------+
# | layers.py                                                             |
# | This module implements layer functions                                |
# +-----------------------------------------------------------------------+

from . import module
from math import sqrt
import torch
torch.set_grad_enabled(False)


# fully connected layer
class fully_connected(module.Module):

    # initialise using dimensions
    def __init__(self, d_in, d_out):
        # call super constructor
        super(fully_connected, self).__init__()
        # calculate variance for weight initialization with xavier initialization
        std_weights = sqrt(2/(d_in + d_out))
        # prepare bias and weight vectors
        self.bias = torch.empty(1, d_out).zero_()
        self.weights = torch.empty(d_in, d_out).normal_(std = std_weights)
        # prepare gradient accumulators
        self.dbias = torch.empty(1, d_out).zero_()
        self.dweights = torch.empty(d_in, d_out).zero_()
        # prepare input memory
        self.input_memory = torch.empty(1, d_in).zero_()

    # forward pass
    def forward(self, input):
        self.input_memory = input
        return self.bias + input @ self.weights

    # backward pass
    def backward(self, gradwrtoutput):
        self.dbias += gradwrtoutput
        self.dweights += self.input_memory.transpose(1,0) @ gradwrtoutput
        return gradwrtoutput @ self.weights.transpose(1,0)
        
    # return parameters
    def param(self):
        return {'bias': self.bias, 'weights': self.weights}

    # zero gradient accumulators
    def zerograd(self):
        self.dbias.zero_()
        self.dweights.zero_()
        return

    # take step in desired direction (directions is a list [d_bias, d_weights])
    def gradient_step(self, directions):
        self.bias += directions[0]
        self.weights += directions[1]
        return
        
    # return the current gradient
    def gradient(self):
        return [self.dbias, self.dweights]