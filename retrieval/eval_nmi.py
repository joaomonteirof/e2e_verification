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
from sklearn.metrics.cluster import normalized_mutual_info_score

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Clustering Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	print(args)

	transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)

	labels_list = [x[1] for x in validset]
	pred_list = []

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try :
		dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['n_classes']
	except KeyError as err:
		print("Key Error: {0}".format(err))
		print('\nProbably old cp has no info regarding classifiers arch!\n')
		n_hidden, hidden_size, softmax, n_classes = get_classifier_config_from_cp(ckpt)
		dropout_prob = args.dropout_prob

	if args.model == 'vgg':
		model = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'resnet':
		model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes)

	print(model.load_state_dict(ckpt['model_state'], strict=False))

	if args.cuda:
		device = get_freer_gpu()
		model = model.cuda(device)

	class_center = {}
	class_count = {}
	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		iterator = tqdm(enumerate(labels_list), total=len(labels_list))
		for i, label in iterator:

			example = str(i)

			example_data = validset[i][0].unsqueeze(0)

			if args.cuda:
				example_data = example_data.cuda(device)

			emb = model.forward(example_data)[0].detach()
			mem_embeddings[example] = emb

			try:
				class_center[label] += emb
				class_count[label] += 1
			except KeyError:
				class_center[label] = emb
				class_count[label] = 1

		for k in class_center:
			class_center[k] /= class_count[k]

		for i, label in enumerate(labels_list):
			class_scores = []
			example = str(i)
			for k in class_center:
				class_scores.append( [model.forward_bin(torch.cat([class_center[k], mem_embeddings[example]],1)).squeeze().item(), k] )

			pred_list.append(max(class_scores)[1])

	print('\nScoring done')

	print('\n NMI: {}'.format(normalized_mutual_info_score(labels_list, pred_list)))