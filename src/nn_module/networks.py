# +-----------------------------------------------------------------------+
# | networks.py                                                           |
# | This module implements network functions                              |
# +-----------------------------------------------------------------------+
from . import module
import torch

class sequential(module.Module):
	def __init__(self, layers):
		super(sequential, self).__init__()
		self.layers = layers
	
	def forward(self, input):
		x = input
		for l in self.layers:
			x = l.forward(x)
		return x
		
	def backward(self, gradwrtoutput):
		x = gradwrtoutput
		for l in reversed(self.layers):
			x = l.backward(x)
		
	def param(self):
		params = {}
		for i, l in enumerate(self.layers):
			params['layer'+str(i)] = l.param()
		return params
		
	def zerograd(self):
		for l in self.layers:
			l.zerograd()
		return
	
	def gradient_step(self, stepsize):
		for l in self.layers:
			l.gradient_step(stepsize)
		return