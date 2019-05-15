# +-----------------------------------------------------------------------+
# | layers.py                                                             |
# | This module implements layer functions                                |
# +-----------------------------------------------------------------------+

from . import module
from numpy import sqrt
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

# dropout layer
class dropout(module.Module):

    # initialization
    def __init__(self, p):
        super().__init__()
        # probability of a unit of being dropped out during forward pass
        self.p = min(max(p, 0.0), 0.9)

    # forward pass
    def forward(self, input):
        if self.training_mode:
            # generate Bernoulli for each unit and scale by correct factor
            self.d = 1/(1-self.p) * torch.bernoulli(torch.empty_like(input).fill_(1-self.p))
        else:
            # layer inactive
            self.d = torch.ones_like(input)
        return input * self.d

    # backward pass
    def backward(self, gradwrtoutput):
        return self.d * gradwrtoutput

    # return dropout table
    def param(self):
        return {'dropout': self.d}

    # empty method (provided for compatibility but has no effect on dropout layer)
    def zerograd(self):
        return None

    # empty method (provided for compatibility but has no effect on dropout layer)
    def gradient_step(self, *args):
        return None

    # empty method (provided for compatibility but has no effect on dropout layer)
    def gradient(self):
        return None

