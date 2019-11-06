from __future__ import print_function
import os
import argparse
import torch
import glob
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpoints')
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

cp_list = glob.glob(args.cp_path+'*.pt')

assert len(cp_list)>0, 'No cp found in the given path!'

for cp in cp_list:

	ckpt = torch.load(cp, map_location = lambda storage, loc: storage)
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
		print('lol', cp.split('/')[-1])
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise
