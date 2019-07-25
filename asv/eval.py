import argparse
import numpy as np
import torch
from kaldi_io import read_mat_scp
from sklearn import metrics
import scipy.io as sio
import model as model_
import glob
import pickle
import os
import sys

from utils.utils import *

def prep_feats(data_, min_nb_frames=100):

	features = data_.T

	if features.shape[1]<min_nb_frames:
		mul = int(np.ceil(min_nb_frames/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :min_nb_frames]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--test-data', type=str, default='./data/test/', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default=None, help='Path to trials file. If None, will be created from spk2utt')
	parser.add_argument('--spk2utt', type=str, default=None, metavar='Path', help='Path to spk2utt file. Will be used in case no trials file is provided')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--model', choices=['resnet_stats', 'resnet_mfcc', 'resnet_lstm', 'resnet_small', 'resnet_large', 'TDNN'], default='resnet_lstm', help='Model arch according to input type')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--latent-size', type=int, default=256, metavar='S', help='latent layer dimension (default: 256)')
	parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
	parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()

	if args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)
	elif args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)
	if args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)
	elif args.model == 'resnet_small':
		model = model_.ResNet_small(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)
	elif args.model == 'resnet_large':
		model = model_.ResNet_large(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=1, ncoef=args.ncoef)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	if args.cuda:
		model = model.cuda(device)

	test_data = None

	files_list = glob.glob(args.test_data+'*.scp')

	for file_ in files_list:
		if test_data is None:
			test_data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				test_data[k] = v

	if args.trials_path:
		utterances_enroll, utterances_test, labels = read_trials(args.trials_path)
	else:
		spk2utt = read_spk2utt(args.spk2utt)
		utterances_enroll, utterances_test, labels = create_trials(spk2utt)

	print('\nAll data ready. Start of scoring')

	cos_scores = []
	e2e_scores = []
	out_e2e = []
	out_cos = []
	mem_embeddings = {}

	model.eval()

	with torch.no_grad():

		for i in range(len(labels)):

			enroll_utt = utterances_enroll[i]

			try:
				emb_enroll = mem_embeddings[enroll_utt]
			except KeyError:

				enroll_utt_data = prep_feats(test_data[enroll_utt])

				if args.cuda:
					enroll_utt_data = enroll_utt_data.cuda(device)

				emb_enroll = model.forward(enroll_utt_data).detach()
				mem_embeddings[enroll_utt] = emb_enroll



			test_utt = utterances_test[i]

			try:
				emb_test = mem_embeddings[test_utt]
			except KeyError:

				test_utt_data = prep_feats(test_data[test_utt])

				if args.cuda:
					enroll_utt_data = enroll_utt_data.cuda(device)
					test_utt_data = test_utt_data.cuda(device)

				emb_test = model.forward(test_utt_data).detach()
				mem_embeddings[test_utt] = emb_test

			e2e_scores.append( model.forward_bin(torch.cat([emb_enroll, emb_test],1)).squeeze().item() )
			cos_scores.append( torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item() )

			out_e2e.append([enroll_utt, test_utt, e2e_scores[-1]])
			out_cos.append([enroll_utt, test_utt, cos_scores[-1]])

	print('\nScoring done')

	with open(args.out_path+'e2e_scores.out', 'w') as f:
		for el in out_e2e:
			item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
			f.write("%s" % item)

	with open(args.out_path+'cos_scores.out', 'w') as f:
		for el in out_cos:
			item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
			f.write("%s" % item)

	e2e_scores = np.asarray(e2e_scores)
	cos_scores = np.asarray(cos_scores)
	all_scores = (e2e_scores + 0.5*(cos_scores+1.))*0.5
	labels = np.asarray(labels)

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, e2e_scores)
	print('\nE2E eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
	print('\nCOS eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

	eer, auc, avg_precision, acc, threshold = compute_metrics(labels, all_scores)
	print('\nCOS eval:')
	print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))
