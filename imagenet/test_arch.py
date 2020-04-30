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
parser.add_argument('--emb-size', type=int, default=256, metavar='N', help='Embedding dimension (default: 256)')
parser.add_argument('--rproj-size', type=int, default=-1, metavar='S', help='Random projection size - active if greater than 1')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG19', nh=args.n_hidden, n_h=args.hidden_size, emb_size=args.emb_size, r_proj_size=args.rproj_size)
elif args.model == 'resnet':
	model = resnet.ResNet50(nh=args.n_hidden, n_h=args.hidden_size, emb_size=args.emb_size, r_proj_size=args.rproj_size)
elif args.model == 'densenet':
	model = densenet.DenseNet121(nh=args.n_hidden, n_h=args.hidden_size, emb_size=args.emb_size, r_proj_size=args.rproj_size)

batch = torch.rand(3, 3, 224, 224)

emb, out = model.forward(batch)

print(emb.size(), out.size())

out_layer = model.out_proj(emb)

print(out_layer.size())

emb = torch.cat([emb,emb],1)

print(emb.size())

pred = model.forward_bin(emb)

print(pred.size())

print(pred.squeeze())