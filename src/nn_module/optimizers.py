# +-----------------------------------------------------------------------+
# | optimizers.py                                                         |
# | This module implements optimizers                                     |
# +-----------------------------------------------------------------------+
import numpy as np
from torch import sqrt 
import torch

# Stochastic Gradient Descent
class SGD(object):

    # initialisation
    def __init__(self, model, lossf):
        super(SGD, self).__init__()
        self.model = model
        self.lossf = lossf
        
    # train for a number of epochs
    def train(self, train_input, train_target, nb_epochs, stepsize, verbose = False):
        # get information
        n_samples = train_input.size(0)
        in_dim = train_input.size(1)
        out_dim = train_target.size(1)
        
        losspath = []
        
        # iterate over epochs
        for e in range(nb_epochs):
            # print current epoch loss and store it in losspath
            if verbose:
                print('Epoch {}...'.format(e))
                print(self.lossf.loss(self.model.forward(train_input), train_target))
            losspath.append(self.lossf.loss(self.model.forward(train_input), train_target).item())
            # generate random sample order
            sample_ordering = np.random.permutation([i for i in range(n_samples)])
            # perform sgd
            for s in sample_ordering:
                # retrieve the sample and its target value
                x = train_input[s, :].view(1, in_dim)
                t = train_target[s, :].view(1, out_dim)
                # set the gradient accumulators to zero
                self.model.zerograd()
                # perform the forward pass given the input x
                out = self.model.forward(x)
                # compute the loss between the output and the target
                loss = self.lossf.loss(out, t)
                # calculate the gradient wrt the output ("dloss = gradwrtoutput")
                dloss = self.lossf.dloss(out, t)
                # perform the backward pass given the gradwrtoutput
                self.model.backward(dloss)
                # retrieve the current model gradients
                grads = self.model.gradient()
                # compute and take the "gradient step" for both the bias and the weights
                for i,g in enumerate(grads):
                    if not g==None:
                        grads[i][0] *= - stepsize   # bias 
                        grads[i][1] *= - stepsize   # weights                
                self.model.gradient_step(grads)
        return losspath


# batch Stochastic Gradient Descent
class batchSGD(object):

    # initialisation
    def __init__(self, model, lossf, batchsize=10):
        super(batchSGD, self).__init__()
        self.model = model
        self.lossf = lossf
        self.batchsize = batchsize
        
    # train for a number of epochs
    def train(self, train_input, train_target, nb_epochs, stepsize, verbose = False):
        # get information
        n_samples = train_input.size(0)
        in_dim = train_input.size(1)
        out_dim = train_target.size(1)
        
        losspath = []
        
        # iterate over epochs
        for e in range(nb_epochs):
            # print current epoch loss and store it in losspath
            if verbose:
                print('Epoch {}...'.format(e))
                print(self.lossf.loss(self.model.forward(train_input), train_target))
            losspath.append(self.lossf.loss(self.model.forward(train_input), train_target).item())
            # generate random sample order
            sample_ordering = np.random.permutation([i for i in range(n_samples)])
            # perform batch sgd
            for i in range(0, n_samples, self.batchsize):
                # set the gradient accumulators to zero
                self.model.zerograd()
                for j in range(self.batchsize):
                    # retrieve the sample and its target value
                    s = sample_ordering[i+j]
                    x = train_input[s, :].view(1, in_dim)
                    t = train_target[s, :].view(1, out_dim)
                    # perform the forward pass given the input x
                    out = self.model.forward(x)
                    # compute the loss between the output and the target
                    loss = self.lossf.loss(out, t)
                    # calculate the gradient wrt the output ("dloss = gradwrtoutput")
                    dloss = self.lossf.dloss(out, t)
                    # perform the backward pass given the gradwrtoutput
                    self.model.backward(dloss)
                # retrieve the current model gradients
                grads = self.model.gradient()
                # compute and take the "gradient step" for both the bias and the weights
                for i,g in enumerate(grads):
                    if not g==None:
                        grads[i][0] *= - stepsize/self.batchsize   # bias 
                        grads[i][1] *= - stepsize/self.batchsize   # weights            
                self.model.gradient_step(grads)
        return losspath