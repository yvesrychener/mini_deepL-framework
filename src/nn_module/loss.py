# +-----------------------------------------------------------------------+
# | loss.py                                                               |
# | This module implements loss functions                                 |
# +-----------------------------------------------------------------------+

import torch

# MSE loss function
class MSE(object):
    # calculate the loss
    def loss(self, output, target):
        return torch.pow(output - target, 2).mean()

    # derivative of the loss with respect to network output
    def dloss(self, output, target):
        return 2*(output - target)