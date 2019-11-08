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
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Symmetry check')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
	parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
	parser.add_argument('--out-path', type=str, default=None, metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-histogram', action='store_true', default=False, help='Disables histogram plot')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	labels_list = [x[1] for x in validset]

	if args.model == 'vgg':
		model = vgg.VGG('VGG16', nh=args.n_hidden, n_h=args.hidden_size)
	elif args.model == 'resnet':
		model = resnet.ResNet18(nh=args.n_hidden, n_h=args.hidden_size)
	elif args.model == 'densenet':
		model = densenet.densenet_cifar(nh=args.n_hidden, n_h=args.hidden_size)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
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

	idxs_enroll, idxs_test, labels = create_trials_labels(labels_list)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	scores_dif = []

	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(range(len(labels)), total=len(labels))
		for i in iterator:

			enroll_ex = str(idxs_enroll[i])

			try:
				emb_enroll = mem_embeddings[enroll_ex]
			except KeyError:

				enroll_ex_data = validset[idxs_enroll[i]][0].unsqueeze(0)

				if args.cuda:
					enroll_ex_data = enroll_ex_data.cuda(device)

				emb_enroll = model.forward(enroll_ex_data).detach()
				mem_embeddings[str(idxs_enroll[i])] = emb_enroll

			test_ex = str(idxs_test[i])

			try:
				emb_test = mem_embeddings[test_ex]
			except KeyError:

				test_ex_data = validset[idxs_test[i]][0].unsqueeze(0)

				if args.cuda:
					test_ex_data = test_ex_data.cuda(device)

				emb_test = model.forward(test_ex_data).detach()
				mem_embeddings[str(idxs_test[i])] = emb_test

			scores_dif.append( abs( model.forward_bin(torch.cat([emb_enroll, emb_test],1)).squeeze().item(), model.forward_bin(torch.cat([emb_test, emb_enroll],1)).squeeze().item() ) )

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
		plt.savefig('sym_hist.pdf', bbox_inches='tight')
