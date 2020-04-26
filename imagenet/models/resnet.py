'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, nh, n_h, z_size, sm_type, num_classes=1000, zero_init_residual=False,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None, dropout_prob=0.25, rproj_size=0):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.sm_type = sm_type
		self.n_classes = num_classes
		self.emb_size = z_size
		self.r_proj_size = rproj_size

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.lin_proj = nn.Sequential(nn.Linear(512*block.expansion, self.emb_size))

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=512*block.expansion, output_features=num_classes)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=512*block.expansion, output_features=num_classes)
		else:
			raise NotImplementedError

		self.classifier = self.make_bin_layers(n_in=2*self.emb_size, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		if self.r_proj_size>0:
			projection = nn.Linear(n_in, self.r_proj_size, bias=False)
			with torch.no_grad():
				projection.weight.div_(torch.norm(projection.weight, keepdim=True))

			projection.weight.require_grad=False

			classifier = nn.ModuleList([projection, nn.Linear(self.r_proj_size, h_size), nn.LeakyReLU(0.1)])

		else:
			classifier = nn.ModuleList([nn.Linear(n_in, h_size), nn.LeakyReLU(0.1)])

		for i in range(n_h_layers-1):
			classifier.append(nn.Linear(h_size, h_size))
			classifier.append(nn.LeakyReLU(0.1))

		classifier.append(nn.Dropout(p=dropout_p))
		classifier.append(nn.Linear(h_size, 1))
		classifier.append(nn.Sigmoid())

		return classifier

	def forward(self, x):

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x_out = torch.flatten(x, 1)
		x_emb = self.lin_proj(x_out)

		return x_emb, x_out

	def forward_bin(self, z):

		for l in self.classifier:
			z = l(z)
		
		return z


def ResNet18(nh=1, n_h=512, emb_size=128, dropout_prob=0.25, sm_type='softmax', n_classes=1000, r_proj_size=0):
	return ResNet(BasicBlock, [2,2,2,2], nh=nh, n_h=n_h, z_size=emb_size, sm_type=sm_type, dropout_prob=dropout_prob, num_classes=n_classes, rproj_size=r_proj_size)

def ResNet34(nh=1, n_h=512, emb_size=128, dropout_prob=0.25, sm_type='softmax', n_classes=1000, r_proj_size=0):
	return ResNet(BasicBlock, [3,4,6,3], nh=nh, n_h=n_h, z_size=emb_size, sm_type=sm_type, dropout_prob=dropout_prob, num_classes=n_classes, rproj_size=r_proj_size)

def ResNet50(nh=1, n_h=512, emb_size=128, dropout_prob=0.25, sm_type='softmax', n_classes=1000, r_proj_size=0):
	return ResNet(Bottleneck, [3,4,6,3], nh=nh, n_h=n_h, z_size=emb_size, sm_type=sm_type, dropout_prob=dropout_prob, num_classes=n_classes, rproj_size=r_proj_size)

def ResNet101(nh=1, n_h=512, emb_size=128, dropout_prob=0.25, sm_type='softmax', n_classes=1000, r_proj_size=0):
	return ResNet(Bottleneck, [3,4,23,3], nh=nh, n_h=n_h, z_size=emb_size, sm_type=sm_type, dropout_prob=dropout_prob, num_classes=n_classes, rproj_size=r_proj_size)

def ResNet152(nh=1, n_h=512, emb_size=128, dropout_prob=0.25, sm_type='softmax', n_classes=1000, r_proj_size=0):
	return ResNet(Bottleneck, [3,8,36,3], nh=nh, n_h=n_h, z_size=emb_size, sm_type=sm_type, dropout_prob=dropout_prob, num_classes=n_classes, rproj_size=r_proj_size)
