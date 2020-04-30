import math
import torch
from torch import nn
from scipy.special import binom
import torch.nn.functional as F

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

	def forward(self, embeddings, target):

		w_norm = F.normalize(self.w, p=2, dim=0)

		cos_theta = embeddings.mm(w_norm)
		cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

		phi_theta = cos_theta - self.m

		target_onehot = torch.zeros(embeddings.size(0), w_norm.size(1)).to(embeddings.device)
		target_onehot.scatter_(1, target.view(-1,1), 1)

		logits = self.s*torch.where(target_onehot==1, phi_theta, cos_theta)

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
		return self.w(embeddings)

class LabelSmoothingLoss(nn.Module):
	def __init__(self, label_smoothing, lbl_set_size, dim=1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - label_smoothing
		self.smoothing = label_smoothing
		self.cls = lbl_set_size
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
