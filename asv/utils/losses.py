import math
import torch
from torch import nn
from scipy.special import binom

class AMSoftmax(nn.Module):

	## adapted from https://github.com/Joker316701882/Additive-Margin-Softmax/blob/master/AM_softmax.py

	def __init__(self, input_features, output_features, m=0.35, s=30.0):
		super().__init__()
		self.input_dim = input_features  # number of input features
		self.output_dim = output_features  # number of classes
		self.s = s
		self.m = m

		# Initialize parameters
		self.w = nn.Parameter(torch.FloatTensor(input_features, output_features))

		self.init_parameters()

	def init_parameters(self):
		nn.init.kaiming_normal_(self.w)

	def forward(self, embeddings, target=None):
		assert target is not None

		self.w.to(embeddings.device)

		w_norm = torch.div(self.w, torch.norm(self.w, 2, 0))

		cos_theta = embeddings.mm(self.w)
		cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

		phi_theta = cos_theta - self.m

		target_onehot = torch.zeros(embeddings.size(0), w_norm.size(1)).to(embeddings.device)
		target_onehot.scatter_(1, target.view(-1,1), 1)

		logits = torch.where(target_onehot==1, phi_theta, cos_theta)

		return logits

class Softmax(nn.Module):

	def __init__(self, input_features, output_features):
		super().__init__()

		self.w = nn.Linear(input_features, output_features)

		self.initialize_params()

	def initialize_params(self):

		for layer in self.modules():

			if isinstance(layer, nn.Linear):
				nn.init.kaiming_normal_(layer.weight)
				layer.bias.data.zero_()

	def forward(self, embeddings, *args):
		self.w.to(embeddings.device)
		return self.w(embeddings)
