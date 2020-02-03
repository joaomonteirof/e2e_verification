import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, nh=1, n_h=512, dropout_prob=0.25, sm_type='softmax', n_classes=1000):
		super(VGG, self).__init__()

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = n_classes

		self.features = self._make_layers(cfg[vgg_name])
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.lin_proj = nn.Sequential(nn.Linear(512 * 7 * 7, 1024), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(1024, 512))

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=512, output_features=self.n_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=512, output_features=self.n_classes)
		else:
			raise NotImplementedError

		self.bin_classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

	def forward(self, x):
		features = self.avgpool(self.features(x))
		print(features.size())
		features = features.view(features.size(0), -1)
		features = self.lin_proj(features)

		return features

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))
		classifier.append(nn.Sigmoid())

		return classifier

	def forward_bin(self, z):

		for l in self.bin_classifier:
			z = l(z)
		
		return z
