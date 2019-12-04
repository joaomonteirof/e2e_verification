from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Symmetry check')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--out-path', type=str, default=None, metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError:
		print('\nProbably old cp has no info regarding classifiers arch!\n')
		n_hidden, hidden_size, softmax = get_classifier_config_from_cp(ckpt)
		dropout_prob = args.dropout_prob

	if args.model == 'vgg':
		model = vgg.VGG('VGG16', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet':
		model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'densenet':
		model = densenet.densenet_cifar(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)

	scores_dif = []

	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(itertools.permutations(range(len(validset)), 3), total=len(validset))
		for i, j, k in iterator:

			anchor_ex = str(idxs_enroll[i])

			try:
				emb_anchor = mem_embeddings[anchor_ex]
			except KeyError:

				anchor_ex_data = validset[i][0].unsqueeze(0)

				if args.cuda:
					anchor_ex_data = anchor_ex_data.cuda(device)

				emb_anchor = model.forward(anchor_ex_data).detach()
				mem_embeddings[anchor_ex] = emb_anchor

			a_ex = str(idxs_enroll[j])

			try:
				emb_a = mem_embeddings[a_ex]
			except KeyError:

				a_ex_data = validset[j][0].unsqueeze(0)

				if args.cuda:
					a_ex_data = a_ex_data.cuda(device)

				emb_a = model.forward(a_ex_data).detach()
				mem_embeddings[a_ex] = emb_a

			b_ex = str(idxs_enroll[k])

			try:
				emb_b = mem_embeddings[b_ex]
			except KeyError:

				b_ex_data = validset[k][0].unsqueeze(0)

				if args.cuda:
					b_ex_data = b_ex_data.cuda(device)

				emb_b = model.forward(b_ex_data).detach()
				mem_embeddings[b_ex] = emb_b

			total_dist = model.forward_bin(torch.cat([emb_anchor, emb_a],1)).squeeze().item() + model.forward_bin(torch.cat([emb_anchor, emb_b],1)).squeeze().item()
			local_dist = model.forward_bin(torch.cat([emb_a, emb_b],1)).squeeze().item()

			scores_dif.append( max(local_dist-total_dist, 0.0) )

	print('\nScoring done')

	print('Avg: {}'.format(np.mean(scores_dif)))
	print('Std: {}'.format(np.std(scores_dif)))
	print('Median: {}'.format(np.median(scores_dif)))
	print('Max: {}'.format(np.max(scores_dif)))
	print('Min: {}'.format(np.min(scores_dif)))

	if not args.no_histogram:
		import matplotlib
		matplotlib.use('agg')
		import matplotlib.pyplot as plt
		plt.hist(scores_dif, normed=True, bins=30)
		plt.savefig('triang_hist_cifar.pdf', bbox_inches='tight')
