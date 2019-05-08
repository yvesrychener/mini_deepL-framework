# +-----------------------------------------------------------------------+
# | activations.py                                                        |
# | This module implements activation functions                           |
# +-----------------------------------------------------------------------+

from . import module
import torch

# tanh activation function
class tanh(module.Module):
    def __init__(self):
        super(tanh, self).__init__()
        self.inputmemory = 0
    # forward pass
    def forward(self, input):
        self.inputmemory = input
        return torch.tanh(input)

    # backward pass
    def backward(self, gradwrtoutput):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2) * gradwrtoutput


# relu activation function
class relu(module.Module):
    def __init__(self):
        super(relu, self).__init__()
        self.inputmemory = 0
    # forward pass
    def forward(self, input):
        self.inputmemory = input
        return input*(input>0).float()
    
    # backward pass
    def backward(self, gradwrtoutput):
        return (self.inputmemory>0).float() * gradwrtoutput
