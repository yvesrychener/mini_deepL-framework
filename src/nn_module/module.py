# +-----------------------------------------------------------------------+
# | module.py                                                             |
# | This module implements the basic module class                         |
# +-----------------------------------------------------------------------+

import torch
torch.set_grad_enabled(False)

# basic module
class Module(object):

    # forward pass
    def forward(self, *input):
        raise NotImplementedError

    # backward pass
    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    # return parameters dict
    def param(self):
        return {}

    # zero gradient accumulators
    def zerograd(self):
        return

    # take gradient step
    def gradient_step(self, directions):
        return 
    
    # return the current gradient
    def gradient(self):
        return