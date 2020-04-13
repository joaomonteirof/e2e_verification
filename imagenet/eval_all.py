from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
import glob
import tqdm
from utils import *

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Imagenet - Evaluation of set of cps')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	cp_group = parser.add_mutually_exclusive_group(required=True)
	cp_group.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpoints')
	cp_group.add_argument('--cp-list', type=str, nargs='+', default=[], metavar='Path', help='Paths to trained models')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	validset = datasets.ImageFolder(args.data_path, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

	if args.cp_path is not None:
		cp_list = glob.glob(args.cp_path+'*.pt')
	elif len(args.cp_list) > 0:
		cp_list = args.cp_list

	best_model, best_e2e_eer = None, float('inf')

	for cp in cp_list:

		ckpt = torch.load(cp, map_location = lambda storage, loc: storage)
		try :
			dropout_prob, n_hidden, hidden_size, softmax, n_classes, emb_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['sm_type'], ckpt['n_classes'], ckpt['emb_size']
		except KeyError as err:
			print("Key Error: {0}".format(err))
			print('\nProbably old cp has no info regarding classifiers arch!\n')
			try:
				n_hidden, hidden_size, softmax, n_classes = get_classifier_config_from_cp(ckpt)
				dropout_prob = args.dropout_prob
				emb_size = 350
			except:
				print('\nSkipping cp {}. Could not load it.'.format(cp))
				continue

		if args.model == 'vgg':
			model = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, emb_size=emb_size)
		elif args.model == 'resnet':
			model = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, emb_size=emb_size)
		elif args.model == 'densenet':
			model = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=n_classes, emb_size=emb_size)

		print(model.load_state_dict(ckpt['model_state'], strict=False))

		if args.cuda:
			device = get_freer_gpu()
			model = model.cuda(device)
		else:
			device = torch.device('cpu')

		embeddings = []
		labels = []

		model.eval()

		iterator = tqdm(valid_loader, total=len(valid_loader))

		with torch.no_grad():

			for batch in iterator:

				x, y = batch

				if args.cuda:
					x = x.to(device)

				emb = model.forward(x)[0].detach()

				embeddings.append(emb.detach().cpu())
				labels.append(y)

		embeddings = torch.cat(embeddings, 0)
		labels = list(torch.cat(labels, 0).squeeze().numpy())

		print('\nEmbedding done')

		idxs_enroll, idxs_test, labels = create_trials_labels(labels)
		print('\n{} trials created out of which {} are target trials'.format(len(idxs_enroll), np.sum(labels)))

		cos_scores = []
		e2e_scores = []
		out_e2e = []
		out_cos = []

		mem_embeddings = {}

		model.eval()

		with torch.no_grad():

			iterator = tqdm(range(0, len(labels), args.batch_size), total=len(labels)//args.batch_size+1)
			for i in iterator:

				enroll_ex = idxs_enroll[i:(min(i+args.batch_size, len(labels)))]
				test_ex = idxs_test[i:(min(i+args.batch_size, len(labels)))]

				enroll_emb = embeddings[enroll_ex,:].to(device)
				test_emb = embeddings[test_ex,:].to(device)
				cat_emb = torch.cat([enroll_emb, test_emb], 1)

				dist_e2e = model.forward_bin(cat_emb).squeeze(1)
				dist_cos = torch.nn.functional.cosine_similarity(enroll_emb, test_emb)
					
				for k in range(dist_e2e.size(0)):
					e2e_scores.append( dist_e2e[k].item() )
					cos_scores.append( dist_cos[k].item() )
					out_e2e.append([str(idxs_enroll[i+k]), str(idxs_test[i+k]), e2e_scores[-1]])
					out_cos.append([str(idxs_enroll[i+k]), str(idxs_test[i+k]), cos_scores[-1]])

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
