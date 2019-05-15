# +-----------------------------------------------------------------------+
# | test.py                                                               |
# | Test the proposed network                                             |
# +-----------------------------------------------------------------------+

# Imports
import nn_module as n
import math
import torch
from torch import Tensor
import matplotlib.pyplot as plt

import sys
if sys.version_info[0] == 3:
    from importlib import reload
	
# Disable Autograd
torch.set_grad_enabled(False)

# Function Definitions

def generate_disc_set(nb):
    '''
    Generates the dataset
    parameters:
    nb   : Number of samples to be generated
    returns:
    input, target   : The input data and target label
    '''
    input = Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).view(nb,1)
    return input, target
	
def errors(test_net, train_input, test_input, train_target, test_target):
    '''
    Display the errors
    parameters:
    test_net   : Trained Network
    train_input, test_input, train_target, test_target: Datasets
    action:
    prints errors
    '''
    pred_train = torch.argmax(test_net.forward(train_input), dim = 1)
    pred_test = torch.argmax(test_net.forward(test_input), dim = 1)
    nbr_errors_train = 0
    nbr_errors_test = 0
    for i in range(1000):
        if pred_train[i].int() != train_target[i,1].int(): nbr_errors_train += 1
        if pred_test[i].int() != test_target[i,1].int(): nbr_errors_test += 1
    print('Final training error: {}%'.format(nbr_errors_train/10))
    print('Final test error: {}%'.format(nbr_errors_test/10))
    
    train_loss = lossf.loss(test_net.forward(train_input), train_target).item()
    print('Final training set loss: {:.4f}'.format(train_loss))
    test_loss = lossf.loss(test_net.forward(test_input), test_target).item()
    print('Final test set loss: {:.4f}'.format(test_loss))
    

# if this python file is main, run it    
if __name__ == '__main__':  
    # generate the datasets (train and test)
    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    # normalize the inputs
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # create class labels (2 dimensional output)
    train_target = torch.cat((train_target, 1-train_target), dim=1)
    test_target = torch.cat((test_target, 1-test_target), dim=1)

    # initialise the test network
    test_net_SGD = n.networks.sequential([
        n.layers.fully_connected(2,25),
        n.activations.relu(),
        n.layers.fully_connected(25,25),
        n.activations.relu(),
        n.layers.fully_connected(25,25),
        n.activations.relu(),
        n.layers.fully_connected(25,2)
    ])

    # define loss and optimizer
    lossf = n.loss.MSE()
    optim = n.optimizers.SGD(test_net_SGD, lossf)

    # train the model
    print('Training Model...')
    losspath_SGD = optim.train(train_input, train_target, 20, 1e-2)
    print('Done')

    # display the errors
    errors(test_net_SGD, train_input, test_input, train_target, test_target)