# +-----------------------------------------------------------------------+
# | activations.py                                                        |
# | This module implements activation functions                           |
# +-----------------------------------------------------------------------+

from . import module
import torch

# tanh (sigmoid) activation function
class tanh(module.Module):
	# forward pass
	def forward(self, input):
		return torch.tanh(input)
	
	# backward pass
	def backward(self, gradwrtoutput):
		return 1 - torch.pow(torch.tanh(gradwrtoutput), 2)
			
	
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
		