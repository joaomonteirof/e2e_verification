from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
import glob
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Imagenet - Evaluation of set of cps')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpoints')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--stats', choices=['cars', 'cub', 'sop', 'imagenet'], default='imagenet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.stats=='cars':
		mean, std = [0.4461, 0.4329, 0.4345], [0.2888, 0.2873, 0.2946]
	elif args.stats=='cub':
		mean, std = [0.4782, 0.4925, 0.4418], [0.2330, 0.2296, 0.2647]
	elif args.stats=='sop':
		mean, std = [0.5603, 0.5155, 0.4796], [0.2939, 0.2991, 0.3085]
	elif args.stats=='imagenet':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

	transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	labels_list = [x[1] for x in validset]

	idxs_enroll, idxs_test, labels = create_trials_labels(labels_list)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	cp_list = glob.glob(args.cp_path+'*.pt')

	best_model, best_e2e_eer = None, float('inf')

	for cp in cp_list:

		ckpt = torch.load(cp, map_location = lambda storage, loc: storage)
		try :
			dropout_prob, n_hidden, hidden_size, softmax, n_classes = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['n_classes']
			if 'r_proj_size' in ckpt:
				rproj_size = ckpt['r_proj_size']
			else:
				rproj_size = -1
		except KeyError as err:
			print("Key Error: {0}".format(err))
			print('\nProbably old cp has no info regarding classifiers arch!\n')
			n_hidden, hidden_size, softmax, n_classes = get_classifier_config_from_cp(ckpt)
			dropout_prob = args.dropout_prob
			rproj_size = -1

		if args.model == 'vgg':
			model = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, r_proj_size=rproj_size)
		elif args.model == 'resnet':
			model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, r_proj_size=rproj_size)
		elif args.model == 'densenet':
			model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, r_proj_size=rproj_size)

		if ckpt['sm_type'] == 'am_softmax':
			del(ckpt['model_state']['out_proj.w'])
		elif ckpt['sm_type'] == 'softmax':
			del(ckpt['model_state']['out_proj.w.weight'])
			del(ckpt['model_state']['out_proj.w.bias'])

		print(model.load_state_dict(ckpt['model_state'], strict=False))

		model.classifier = model.classifier[:-1]
		model.eval()

		if args.cuda:
			device = get_freer_gpu()
			model = model.cuda(device)

		cos_scores = []
		e2e_scores = []
		out_e2e = []
		out_cos = []

		mem_embeddings = {}

		model.eval()

		with torch.no_grad():

			for i in range(len(labels)):

				enroll_ex = str(idxs_enroll[i])

				try:
					emb_enroll = mem_embeddings[enroll_ex]
				except KeyError:

					enroll_ex_data = validset[idxs_enroll[i]][0].unsqueeze(0)

					if args.cuda:
						enroll_ex_data = enroll_ex_data.cuda(device)

					emb_enroll = model.forward(enroll_ex_data)[0].detach()
					mem_embeddings[str(idxs_enroll[i])] = emb_enroll

				test_ex = str(idxs_test[i])

				try:
					emb_test = mem_embeddings[test_ex]
				except KeyError:

					test_ex_data = validset[idxs_test[i]][0].unsqueeze(0)

					if args.cuda:
						test_ex_data = test_ex_data.cuda(device)

					emb_test = model.forward(test_ex_data)[0].detach()
					mem_embeddings[str(idxs_test[i])] = emb_test

				e2e_scores.append( model.forward_bin(torch.cat([emb_enroll, emb_test],1)).squeeze().item() )
				cos_scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )

				out_e2e.append([str(idxs_enroll[i]), str(idxs_test[i]), e2e_scores[-1]])
				out_cos.append([str(idxs_enroll[i]), str(idxs_test[i]), cos_scores[-1]])

		e2e_scores = np.asarray(e2e_scores)
		cos_scores = np.asarray(cos_scores)
		all_scores = ((torch.sigmoid(torch.from_numpy(e2e_scores).float()) + 0.5*(torch.from_numpy(cos_scores)+1.))*0.5).numpy()
		labels = np.asarray(labels)
		model_id = cp.split('/')[-1]

		print('\nEval of model {}:'.format(model_id))

		e2e_eer, e2e_auc, avg_precision, acc, threshold = compute_metrics(labels, e2e_scores)
		print('\nE2E:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(e2e_eer, e2e_auc, avg_precision, acc, threshold))

		cos_eer, cos_auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
		print('\nCOS:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(cos_eer, cos_auc, avg_precision, acc, threshold))

		fus_eer, fus_auc, avg_precision, acc, threshold = compute_metrics(labels, all_scores)
		print('\nFUS:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(fus_eer, fus_auc, avg_precision, acc, threshold))

		if e2e_eer<best_e2e_eer:
			best_model, best_e2e_eer, best_e2e_auc, best_cos_eer, best_cos_auc, best_fus_eer, best_fus_auc = model_id, e2e_eer, e2e_auc, cos_eer, cos_auc, fus_eer, fus_auc

	print('Best model and corresponding E2E eer and auc: {} - {} - {}'.format(best_model, best_e2e_eer, best_e2e_auc))
	print('Corresponding COS eer and auc: {} - {} - {}'.format(best_model, best_cos_eer, best_cos_auc))
	print('Corresponding FUS eer and auc: {} - {} - {}'.format(best_model, best_fus_eer, best_fus_auc))
