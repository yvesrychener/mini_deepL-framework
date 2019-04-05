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
	# forward pass
	def forward(self, input):
		return input*(input>0).float()
	
	# backward pass
	def backward(self, gradwrtoutput):
		return (gradwrtoutput>0).float()
		