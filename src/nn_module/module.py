# +-----------------------------------------------------------------------+
# | module.py                                                             |
# | This module implements the basic module class                         |
# +-----------------------------------------------------------------------+

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
    def gradient_step(self, stepsize):
        return 