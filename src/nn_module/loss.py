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

# Cross-Entropy loss function
class CE(object):
    # Calculate the mean cross-entropy loss
    def loss(self, output, target):
        return -torch.log(
            output[:, target.argmax(dim=1)].exp() / output.exp().sum(dim=1)
        ).mean()

    # derivative of the loss with respect to network output
    def dloss(self, output, target):
        var = -(
            torch.exp(output - output[:, target.argmax(dim=1)]).sum()
            * torch.exp(output - output[:, target.argmax(dim=1)])
        )
        var[:, target.argmax(dim=1)] = 0
        return var

