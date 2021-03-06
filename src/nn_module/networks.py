# +-----------------------------------------------------------------------+
# | networks.py                                                           |
# | This module implements network functions                              |
# +-----------------------------------------------------------------------+
from . import module
import torch
torch.set_grad_enabled(False)

class sequential(module.Module):
    # initialize
    def __init__(self, layers):
        super(sequential, self).__init__()
        self.layers = layers

    # forward pass
    def forward(self, input):
        x = input
        # pass input through all the layers
        for l in self.layers:
            x = l.forward(x)
        return x

    # backward pass
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        # pass gradient through all the layers in reverse order
        for l in reversed(self.layers):
            x = l.backward(x)

    # return parameters
    def param(self):
        params = {}
        layer_n = 0
        for i, l in enumerate(self.layers):
            # return all the layer parameters, ignore empty parameter dicts (-> activations & similar)
            layer_param = l.param()
            if bool(layer_param):
                params['layer'+str(layer_n)] = layer_param
                layer_n +=1
        return params
        
    # zero gradient
    def zerograd(self):
        for l in self.layers:
            l.zerograd()
        return

    # take gradient step
    def gradient_step(self, directions):
        for i, l in enumerate(self.layers):
            l.gradient_step(directions[i])
        return
    
    # return the current gradient
    def gradient(self):
        gradients = []
        for l in self.layers:
            gradients.append(l.gradient())
        return gradients
        
    # allows to handle train and test mode required for dropout layers
    def set_training_mode(self, mode):
        print('Switching training mode to {} for all layers...'.format(mode))
        for l in self.layers:
            l.set_training_mode(mode)
