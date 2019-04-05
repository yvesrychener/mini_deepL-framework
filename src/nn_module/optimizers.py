# +-----------------------------------------------------------------------+
# | optimizers.py                                                         |
# | This module implements optimizers                                     |
# +-----------------------------------------------------------------------+
import numpy as np

# Stochastic Gradient Descent
class SGD(object):
	
	# initialisation
	def __init__(self, model, lossf):
		super(SGD, self).__init__()
		self.model = model
		self.lossf = lossf
		
	# train for a number of epochs
	def train(self, train_input, train_target, nb_epochs, stepsize):
		# get information
		n_samples = train_input.size(1)
		in_dim = train_input.size(0)
		out_dim = train_target.size(0)
		
		# iterate over epochs
		for e in range(nb_epochs):
			# print current epoch
			print('Epoch {}...'.format(e))
			# generate random sample order
			sample_ordering = np.random.permutation([i for i in range(n_samples)])
			# perform sgd
			for s in sample_ordering:
				x = train_input[:, s].view(in_dim, 1)
				t = train_target[:, s].view(out_dim, 1)
				self.model.zerograd()
				out = self.model.forward(x)
				loss = self.lossf.loss(out, t)
				dloss = self.lossf.dloss(out, t)
				self.model.backward(dloss)
				self.model.gradient_step(stepsize)