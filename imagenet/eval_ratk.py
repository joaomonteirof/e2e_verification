from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Retrieval Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--k-list', nargs='+', required=True, help='List of k values for R@K computation')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
	if type(args.k_list[0]) is str:
		args.k_list = [int(x) for x in args.k_list[0].split(',')]

	print(args)

	transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)

	labels_list = [x[1] for x in validset]
	r_at_k_e2e = {'R@'+str(x):0 for x in args.k_list}
	r_at_k_cos = {'R@'+str(x):0 for x in args.k_list}

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
	except KeyError as err:
		print("Key Error: {0}".format(err))
		print('\nProbably old cp has no info regarding classifiers arch!\n')
		n_hidden, hidden_size, softmax = get_classifier_config_from_cp(ckpt)
		dropout_prob = args.dropout_prob

	if args.model == 'vgg':
		model = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet':
		model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)

	print(model.load_state_dict(ckpt['model_state'], strict=False))

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)

	cos_scores = {}
	e2e_scores = {}
	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(enumerate(labels_list), total=len(labels_list))
		for i, label_1 in iterator:

			enroll_ex = str(i)

			try:
				emb_enroll = mem_embeddings[enroll_ex]

			except KeyError:

				enroll_ex_data = validset[i][0].unsqueeze(0)

				if args.cuda:
					enroll_ex_data = enroll_ex_data.cuda(device)

				emb_enroll = model.forward(enroll_ex_data).detach()
				mem_embeddings[enroll_ex] = emb_enroll

			e2e_scores[enroll_ex] = []
			cos_scores[enroll_ex] = []

			for j, label_2 in enumerate(labels_list):
				
				if i==j: continue ## skip same example

				test_ex = str(j)

				try:
					emb_test = mem_embeddings[test_ex]

				except KeyError:

					test_ex_data = validset[j][0].unsqueeze(0)

					if args.cuda:
						test_ex_data = test_ex_data.cuda(device)

					emb_test = model.forward(test_ex_data).detach()
					mem_embeddings[test_ex] = emb_test

				e2e_scores[enroll_ex].append( [model.forward_bin(torch.cat([emb_enroll, emb_test],1)).squeeze().item(), label_2] )
				cos_scores[enroll_ex].append( [torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item(), label_2] )

	print('\nScoring done')

for i, label in enumerate(labels_list):
	eval_ex = str(i)
	sorted_e2e_classes = np.array(sorted(e2e_scores[eval_ex], reverse=True))[:,1]
	sorted_cos_classes = np.array(sorted(cos_scores[eval_ex], reverse=True))[:,1]

	for k in args.k_list:
		if label in sorted_e2e_classes[:k]:
			r_at_k_e2e['R@'+str(k)]+=1
		if label in sorted_cos_classes[:k]:
			r_at_k_e2e['R@'+str(k)]+=1

print('\nR@k - E2E:')
print(r_at_k_e2e)
print('\nR@k - Cos:')
print(r_at_k_cos)
print('\n')