import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.losses import AMSoftmax, Softmax

class SelfAttention(nn.Module):
	def __init__(self, hidden_size):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

		init.kaiming_uniform_(self.att_weights)

	def forward(self, inputs):

		batch_size = inputs.size(0)
		weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

		attentions = F.softmax(torch.tanh(weights.squeeze(2)),dim=1)
		weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		noise = 1e-5*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.to(inputs.get_device())

		avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		return representations

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes))

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential( nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes) )

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out


class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out

class ResNet_stats(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, layers=[3,4,6,3], block=PreActBottleneck, proj_size=100, ncoef=23, dropout_prob=0.25, sm_type='softmax', ndiscriminators=1, r_proj_size=0):
		self.in_planes = 32
		super(ResNet_stats, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef
	
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
		else:
			raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)
		x = torch.cat([x.mean(-1), x.std(-1)], dim=1)

		fc = F.relu(self.lbn(self.fc(x)))
		mu = self.fc_mu(fc)

		return mu, fc

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

class ResNet_mfcc(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, layers=[3,4,6,3], block=PreActBottleneck, proj_size=100, ncoef=23, dropout_prob=0.25, sm_type='softmax', ndiscriminators=1, r_proj_size=0):
		self.in_planes = 32
		super(ResNet_mfcc, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef

		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		self.attention = SelfAttention(block.expansion*512)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
		else:
			raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

class ResNet_lstm(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, layers=[3,4,6,3], block=PreActBottleneck, proj_size=100, ncoef=23, dropout_prob=0.25, sm_type='softmax', ndiscriminators=1, r_proj_size=0):
		self.in_planes = 32
		super(ResNet_lstm, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef
	
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.lstm = nn.LSTM(block.expansion*512, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(2*512+256,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		self.attention = SelfAttention(512)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
		else:
			raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2).permute(2,0,1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = torch.zeros(2*2, batch_size, 256)
		c0 = torch.zeros(2*2, batch_size, 256)

		if x.is_cuda:
			h0 = h0.to(x.get_device())
			c0 = c0.to(x.get_device())

		out_seq, (h_, c_) = self.lstm(x, (h0, c0))

		stats = self.attention(out_seq.permute(1,0,2).contiguous())

		x = torch.cat([stats,h_.mean(0)],dim=1)

		fc = F.relu(self.lbn(self.fc(x)))
		emb = self.fc_mu(fc)
		return emb, fc

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

class ResNet_small(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, layers=[2,2,2,2], block=PreActBlock, proj_size=0, ncoef=23, dropout_prob=0.25, sm_type='none', ndiscriminators=1, r_proj_size=0):
		self.in_planes = 32
		super(ResNet_small, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef

		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		self.attention = SelfAttention(block.expansion*512)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

class ResNet_large(nn.Module):
	def __init__(self, n_z=256, nh=1, n_h=512, layers=[3,4,23,3], block=PreActBottleneck, proj_size=100, ncoef=23, dropout_prob=0.25, sm_type='softmax', ndiscriminators=1, r_proj_size=0):
		self.in_planes = 32
		super(ResNet_large, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef

		self.conv1 = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc = nn.Linear(block.expansion*512*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_mu = nn.Linear(512, n_z)

		self.initialize_params()

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		self.attention = SelfAttention(block.expansion*512)

		if sm_type=='softmax':
			self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
		elif sm_type=='am_softmax':
			self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
		else:
			raise NotImplementedError

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.relu(self.lbn(self.fc(stats)))

		mu = self.fc_mu(fc)
		return mu, fc

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		noise = torch.rand(x.size()).to(x.device)*1e-6
		x = x + noise 
		mu = x.mean(dim=2, keepdim=True)
		std = x.std(dim=2, keepdim=True)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	# Architecture taken from https://github.com/santi-pdp/pase/blob/master/pase/models/tdnn.py
	def __init__(self, n_z=256, nh=1, n_h=512, proj_size=0, ncoef=23, sm_type='none', dropout_prob=0.25, ndiscriminators=1, r_proj_size=0):
		super(TDNN, self).__init__()

		self.ndiscriminators = ndiscriminators
		self.r_proj_size = r_proj_size
		self.classifier = nn.ModuleList()
		self.dropout_prob = dropout_prob
		self.n_hidden = nh
		self.hidden_size = n_h
		self.latent_size = n_z
		self.sm_type = sm_type
		self.ncoef = ncoef

		self.model = nn.Sequential( nn.Conv1d(ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 5, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 7),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = StatisticalPooling()

		self.post_pooling_1 = nn.Sequential(nn.Conv1d(3000, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True) )

		self.post_pooling_2 = nn.Sequential(nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, n_z, 1) )

		if ndiscriminators>1:
			for i in range(self.ndiscriminators):
				self.classifier.append(self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob))
		else:
			self.classifier = self.make_bin_layers(n_in=2*512, n_h_layers=nh, h_size=n_h, dropout_p=dropout_prob)

		if proj_size>0 and sm_type!='none':
			if sm_type=='softmax':
				self.out_proj=Softmax(input_features=n_z, output_features=proj_size)
			elif sm_type=='am_softmax':
				self.out_proj=AMSoftmax(input_features=n_z, output_features=proj_size)
			else:
				raise NotImplementedError

		# get output features at affine after stats pooling
		# self.model = nn.Sequential(*list(self.model.children())[:-5])

	def make_bin_layers(self, n_in, n_h_layers, h_size, dropout_p):

		if self.r_proj_size>0:
			projection = nn.Linear(n_in, self.r_proj_size, bias=False)
			with torch.no_grad():
				projection.weight /= torch.norm(projection.weight.squeeze()).item()

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
		x = self.model(x.squeeze(1))
		x = self.pooling(x)
		fc = self.post_pooling_1(x)
		x = self.post_pooling_2(fc)

		return x.squeeze(-1), fc.squeeze(-1)

	def forward_bin(self, z):

		if self.ndiscriminators>1:
			out = []
			for disc in self.classifier:
				z_ = z
				for l in disc:
					z_ = l(z_)
				out.append(z_)

			return out

		else:
			for l in self.classifier:
				z = l(z)
		
			return z

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv1d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
