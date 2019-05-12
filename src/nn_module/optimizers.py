# +-----------------------------------------------------------------------+
# | optimizers.py                                                         |
# | This module implements optimizers                                     |
# +-----------------------------------------------------------------------+
import numpy as np
import torch
from torch import sqrt
import copy
torch.set_grad_enabled(False)

# OPTIMIZER 1: Stochastic Gradient Descent
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


# OPTIMIZER 2: batch Stochastic Gradient Descent
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
            for k in range(0, n_samples, self.batchsize):
                # set the gradient accumulators to zero
                self.model.zerograd()
                for j in range(self.batchsize):
                    # retrieve the sample and its target value
                    s = sample_ordering[k+j]
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


# OPTIMIZER 3: AdaGrad
class AdaGrad(object):

    # initialisation
    def __init__(self, model, lossf, gamma=0.01, delta=1e-8, batchsize=10):
        super(AdaGrad, self).__init__()
        self.model = model
        self.lossf = lossf
        self.batchsize = batchsize
        # global learning rate
        self.gamma = gamma
        # damping coefficient
        self.delta = delta
        
    # train for a number of epochs
    def train(self, train_input, train_target, nb_epochs, verbose = False):
        # get information
        n_samples = train_input.size(0)
        in_dim = train_input.size(1)
        out_dim = train_target.size(1)
        # initialize parameters
        losspath = []
        r = None
        # iterate over epochs
        for e in range(nb_epochs):
            # print current epoch loss and store it in losspath
            if verbose:
                print('Epoch {}...'.format(e))
                print(self.lossf.loss(self.model.forward(train_input), train_target).item())
            losspath.append(self.lossf.loss(self.model.forward(train_input), train_target).item())
            # generate random sample order
            sample_ordering = np.random.permutation([i for i in range(n_samples)])
            # iterate over minibatches
            for k in range(0, n_samples, self.batchsize):
                # set the gradient accumulators to zero
                self.model.zerograd()
                for j in range(self.batchsize):
                    # retrieve the sample and its target value
                    s = sample_ordering[k+j]
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
                # initialize r with the correct shape
                if r is None:
                    r = copy.deepcopy(grads)
                # calculate gradient update
                for i,g in enumerate(grads):
                    # to ignore the activation layers (where g = None)
                    if not g==None:
                        # initialize the parameter in the first iteration (first minibatch in first epoch)
                        if e==0 and k==0:
                            r[i][0] = grads[i][0]*grads[i][0]
                            r[i][1] = grads[i][1]*grads[i][1]
                        # update parameter
                        else:
                            r[i][0] = r[i][0] + grads[i][0]*grads[i][0]
                            r[i][1] = r[i][1] + grads[i][1]*grads[i][1]
                        # calculate the gradient step
                        grads[i][0] = - (self.gamma/(self.delta+sqrt(r[i][0])))*grads[i][0]   # bias 
                        grads[i][1] = - (self.gamma/(self.delta+sqrt(r[i][0])))*grads[i][1]   # weights            
                self.model.gradient_step(grads)
        return losspath


# Optimizer 4: RMSProp
class RMSProp(object):

    # initialisation
    def __init__(self, model, lossf, gamma=0.01, delta=1e-8, tau=0.9, batchsize=10):
        super(RMSProp, self).__init__()
        self.model = model
        self.lossf = lossf
        self.batchsize = batchsize
        # global learning rate
        self.gamma = gamma
        # damping coefficient
        self.delta = delta
        # decaying parameter
        self.tau = tau
        
    # train for a number of epochs
    def train(self, train_input, train_target, nb_epochs, verbose = False):
        # get information
        n_samples = train_input.size(0)
        in_dim = train_input.size(1)
        out_dim = train_target.size(1)
        # initialize parameters
        losspath = []
        r = None
        # iterate over epochs
        for e in range(nb_epochs):
            # print current epoch loss and store it in losspath
            if verbose:
                print('Epoch {}...'.format(e))
                print(self.lossf.loss(self.model.forward(train_input), train_target).item())
            losspath.append(self.lossf.loss(self.model.forward(train_input), train_target).item())
            # generate random sample order
            sample_ordering = np.random.permutation([i for i in range(n_samples)])
            # iterate over mini-batches
            for k in range(0, n_samples, self.batchsize):
                # set the gradient accumulators to zero
                self.model.zerograd()
                for j in range(self.batchsize):
                    # retrieve the sample and its target value
                    s = sample_ordering[k+j]
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
                # initialize r with correct shape
                if r is None:
                    r = copy.deepcopy(grads)
                # compute gradient update
                for i,g in enumerate(grads):
                    # to ignore the activation layers (where g = None)
                    if not g==None:
                        # initialize the parameter in the first iteration (first minibatch in first epoch)
                        if e==0 and k==0:
                            r[i][0] = (1-self.tau)*grads[i][0]*grads[i][0]
                            r[i][1] = (1-self.tau)*grads[i][1]*grads[i][1]
                        # update parameter
                        else:
                            r[i][0] = self.tau*r[i][0] + (1-self.tau)*grads[i][0]*grads[i][0]
                            r[i][1] = self.tau*r[i][1] + (1-self.tau)*grads[i][1]*grads[i][1]
                        # calculate gradient step
                        grads[i][0] = - (self.gamma/(self.delta+sqrt(r[i][0])))*grads[i][0]   # bias 
                        grads[i][1] = - (self.gamma/(self.delta+sqrt(r[i][0])))*grads[i][1]   # weights            
                self.model.gradient_step(grads)
        return losspath


# OPTIMIZER 5: Adam
class Adam(object):
    # initialisation
    def __init__(self, model, lossf, gamma=0.001, delta=1e-8, beta1=0.9, beta2=0.999, batchsize=10):
        super(Adam, self).__init__()
        self.model = model
        self.lossf = lossf
        self.batchsize = batchsize
        # global learning rate
        self.gamma = gamma
        # damping coefficient
        self.delta = delta
        # first order decaying parameter
        self.beta1 = beta1
        # second order decaying parameter
        self.beta2 = beta2
        
    # train for a number of epochs
    def train(self, train_input, train_target, nb_epochs, verbose = False):
        # get information
        n_samples = train_input.size(0)
        in_dim = train_input.size(1)
        out_dim = train_target.size(1)
        # initialize vectors
        losspath = []
        m1 = None
        m1_hat = None
        m2 = None
        m2_hat = None
        ctr = 0
        # iterate over epochs
        for e in range(nb_epochs):
            # print current epoch loss and store it in losspath
            if verbose:
                print('Epoch {}...'.format(e))
                print(self.lossf.loss(self.model.forward(train_input), train_target).item())
            losspath.append(self.lossf.loss(self.model.forward(train_input), train_target).item())
            # generate random sample order
            sample_ordering = np.random.permutation([i for i in range(n_samples)])
            for k in range(0, n_samples, self.batchsize):
                # set the gradient accumulators to zero
                self.model.zerograd()
                # iterate over mini-batch
                for j in range(self.batchsize):
                    # retrieve the sample and its target value
                    s = sample_ordering[k+j]
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
                # retrieve the current model gradients (returns a list of grads for each layer)
                grads = self.model.gradient()
                # to initialize proper list shape in first iteration
                if m1 is None:
                    m1 = copy.deepcopy(grads)
                    m1_hat = copy.deepcopy(grads)
                if m2 is None:
                    m2 = copy.deepcopy(grads)
                    m2_hat = copy.deepcopy(grads)
                # update iteration counter
                ctr += 1
                # calculate gradient update
                for i,g in enumerate(grads):
                    # to ignore the activation layers (where g = None)
                    if not g==None:
                        # initialize the parameters in the first iteration
                        if ctr==1:
                            m1[i][0] = (grads[i][0]/self.batchsize)*(1-self.beta1)
                            m1[i][1] = (grads[i][1]/self.batchsize)*(1-self.beta1)
                            m2[i][0] = (grads[i][0]/self.batchsize).pow(2)*(1-self.beta2)
                            m2[i][1] = (grads[i][1]/self.batchsize).pow(2)*(1-self.beta2)
                        # update the parameters
                        else:
                            m1[i][0] = m1[i][0]*self.beta1 + grads[i][0]*(1-self.beta1)
                            m1[i][1] = m1[i][1]*self.beta1 + grads[i][1]*(1-self.beta1)
                            m2[i][0] = m2[i][0]*self.beta2 + grads[i][0].pow(2)*(1-self.beta2)
                            m2[i][1] = m2[i][1]*self.beta2 + grads[i][1].pow(2)*(1-self.beta2)
                        # correct bias
                        m1_hat[i][0] = m1[i][0]/(1-self.beta1**ctr)
                        m1_hat[i][1] = m1[i][1]/(1-self.beta1**ctr)
                        m2_hat[i][0] = m2[i][0]/(1-self.beta2**ctr)
                        m2_hat[i][1] = m2[i][1]/(1-self.beta2**ctr)
                        # gradient step update
                        grads[i][0] = - self.gamma*m1_hat[i][0]/(self.delta+sqrt(m2_hat[i][0]))  # bias 
                        grads[i][1] = - self.gamma*m1_hat[i][1]/(self.delta+sqrt(m2_hat[i][1]))  # weights
                self.model.gradient_step(grads)

        return losspath