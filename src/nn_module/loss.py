# +-----------------------------------------------------------------------+
# | loss.py                                                               |
# | This module implements loss functions                                 |
# +-----------------------------------------------------------------------+

import torch
torch.set_grad_enabled(False)

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
            output[range(output.size(0)), target.argmax(dim=1)].exp()
            /
            output.exp().sum(dim=1).view(-1,1)
        ).mean()

    # derivative of the loss with respect to network output
    def dloss(self, output, target):
        var = output.exp() / output.exp().sum(dim=1).view(-1, 1)
        var[range(output.size(0)), target.argmax(dim=1)] -= 1
        return var.mean(dim=0).view(1,-1)

