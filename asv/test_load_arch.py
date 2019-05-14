from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_

def print_model_norms_sum(model_):
	norm_sum = 0.0
	for params in list(model_.parameters()):
		norm_sum+=params.norm(2).item()
	print('Sum of norms: {}'.format(norm_sum))

def print_norms_diff(model_1, model_2):
	for i, params in enumerate(zip(list(model_1.parameters()), list(model_2.parameters()))):
		norm_diff=params[0].norm(2).item()-params[1].norm(2).item()
		print('Diff of norms in layer {}: {}'.format(i, norm_diff))

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['resnet_lstm'], default='resnet_lstm', help='Model arch according to input type')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--latent-size', type=int, default=256, metavar='S', help='latent layer dimension (default: 256)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--pairwise', action='store_true', default=False, help='Enables layer-wise comparison of norms')
args = parser.parse_args()

if args.model == 'resnet_lstm':
	model = model_.ResNet_lstm(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)

if args.pairwise:
	if args.model == 'resnet_lstm':
		clone_model = model_.ResNet_lstm(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)

	clone_model.load_state_dict(model.state_dict(), strict=True)

print_model_norms_sum(model)

print(' ')

ckpt = torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)

try:
	model.load_state_dict(ckpt['model_state'], strict=True)
except RuntimeError as err:
	print("Runtime Error: {0}".format(err))
except:
	print("Unexpected error:", sys.exc_info()[0])
	raise

print(' ')

print_model_norms_sum(model)

print(' ')

if args.pairwise:
	print_norms_diff(clone_model, model)
