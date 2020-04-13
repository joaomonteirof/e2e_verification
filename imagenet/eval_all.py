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
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	cp_group = parser.add_mutually_exclusive_group(required=True)
	cp_group.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpoints')
	cp_group.add_argument('--cp-list', type=str, nargs='+', default=[], metavar='Path', help='Paths to trained models')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	labels_list = [x[1] for x in validset]

	idxs_enroll, idxs_test, labels = create_trials_labels(labels_list)
	print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

	if args.cp_path is not None:
		cp_list = glob.glob(args.cp_path+'*.pt')
	elif len(args.cp_list) > 0:
		cp_list = args.cp_list

	best_model, best_e2e_eer = None, float('inf')

	for cp in cp_list:

		ckpt = torch.load(cp, map_location = lambda storage, loc: storage)
		try :
			dropout_prob, n_hidden, hidden_size, softmax, emb_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type']
		except KeyError as err:
			print("Key Error: {0}".format(err))
			print('\nProbably old cp has no info regarding classifiers arch!\n')
			try:
				n_hidden, hidden_size, softmax = get_classifier_config_from_cp(ckpt)
				dropout_prob = args.dropout_prob
			except:
				print('\nSkipping cp {}. Could not load it.'.format(cp))
				continue

		if args.model == 'vgg':
			model = vgg.VGG('VGG16', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, emb_size=emb_size)
		elif args.model == 'resnet':
			model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, emb_size=emb_size)
		elif args.model == 'densenet':
			model = densenet.densenet_cifar(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, emb_size=emb_size)

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

				e2e_scores.append( model.forward_bin(torch.cat([emb_enroll, emb_test],1)).squeeze().item() )
				cos_scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )

				out_e2e.append([str(idxs_enroll[i]), str(idxs_test[i]), e2e_scores[-1]])
				out_cos.append([str(idxs_enroll[i]), str(idxs_test[i]), cos_scores[-1]])

		e2e_scores = np.asarray(e2e_scores)
		cos_scores = np.asarray(cos_scores)
		all_scores = (e2e_scores + 0.5*(cos_scores+1.))*0.5
		labels = np.asarray(labels)

		print('\nEval of model {}:'.format(cp))

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
			best_model, best_e2e_eer, best_e2e_auc, best_cos_eer, best_cos_auc, best_fus_eer, best_fus_auc = cp, e2e_eer, e2e_auc, cos_eer, cos_auc, fus_eer, fus_auc

	print('Best model and corresponding E2E eer and auc: {} - {} - {}'.format(best_model, best_e2e_eer, best_e2e_auc))
	print('Corresponding COS eer and auc: {} - {} - {}'.format(best_model, best_cos_eer, best_cos_auc))
	print('Corresponding FUS eer and auc: {} - {} - {}'.format(best_model, best_fus_eer, best_fus_auc))
