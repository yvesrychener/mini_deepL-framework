# +-----------------------------------------------------------------------+
# | layers.py                                                             |
# | This module implements layer functions                                |
# +-----------------------------------------------------------------------+

from . import module
import torch


# fully connected layer
class fully_connected(module.Module):
	
	# initialise using dimensions
	def __init__(self, d_in, d_out):
		# call super constructor
		super(fully_connected, self).__init__()
		# prepare bias and weight vectors
		self.bias = torch.empty(d_out, 1).normal_()
		self.weights = torch.empty(d_out, d_in).normal_()
		# prepare gradient accumulators
		self.dbias = torch.empty(d_out, 1).zero_()
		self.dweights = torch.empty(d_out, d_in).zero_()
		# prepare input memory
		self.input_memory = torch.empty(d_in, 1).zero_()
	
	# forward pass
	def forward(self, input):
		self.input_memory = input
		return self.bias + self.weights @ input
		
	# backward pass
	def backward(self, gradwrtoutput):
		self.dbias += gradwrtoutput
		self.dweights += gradwrtoutput @ self.input_memory.transpose(1,0)
		return self.weights.transpose(1,0) @ gradwrtoutput
		
	# return parameters
	def param(self):
		return {'bias': self.bias, 'weights': self.weights}
		
	# zero gradient accumulators
	def zerograd(self):
		self.dbias.zero_()
		self.dweights.zero_()
		return
	
	# take gradient step with desired stepsize
	def gradient_step(self, stepsize):
		self.bias -= stepsize * self.dbias
		self.weights -= stepsize * self.dweights
		return