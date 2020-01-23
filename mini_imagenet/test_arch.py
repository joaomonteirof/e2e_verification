from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG16', nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'resnet':
	model = resnet.ResNet18(nh=args.n_hidden, n_h=args.hidden_size)
elif args.model == 'densenet':
	model = densenet.densenet_cifar(nh=args.n_hidden, n_h=args.hidden_size)

batch = torch.rand(3, 3, 84, 84)

emb = model.forward(batch)

print(emb.size())

out = model.out_proj(emb)

print(out.size())

emb = torch.cat([emb,emb],1)

print(emb.size())

pred = model.forward_bin(emb)

print(pred.size())

scores_p = model.forward_bin(emb).squeeze()

print(scores_p.size())
