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
import pathlib

from utils.utils import *

def prep_feats(data_, min_nb_frames=50):

	features = data_.T

	if features.shape[1]<min_nb_frames:
		mul = int(np.ceil(min_nb_frames/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :min_nb_frames]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--enroll-data', type=str, default='./data/enroll/', metavar='Path', help='Path to input data')
	parser.add_argument('--test-data', type=str, default='./data/test/', metavar='Path', help='Path to input data')
	parser.add_argument('--unlab-data', type=str, default=None, metavar='Path', help='Path to unlabeled data for centering')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--spk2utt', type=str, default='./data/spk2utt', metavar='Path', help='Path to enrollment spk2utt file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
	parser.add_argument('--model', choices=['resnet_stats', 'resnet_mfcc', 'resnet_lstm', 'resnet_small', 'resnet_large', 'TDNN'], default='resnet_lstm', help='Model arch according to input type')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path for saving computed scores')
	parser.add_argument('--out-prefix', type=str, default=None, metavar='Path', help='Prefix to be added to score files')
	parser.add_argument('--max-nscores', type=int, default=200, metavar='S', help='Max. number of test scores to consider (default: 200)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--inner', action='store_true', default=True, help='Inner layer as embedding')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cuda:
		device = get_freer_gpu()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	if args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])
	elif args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])
	elif args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])
	elif args.model == 'resnet_small':
		model = model_.ResNet_small(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])
	elif args.model == 'resnet_large':
		model = model_.ResNet_large(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=ckpt['latent_size'], nh=ckpt['n_hidden'], n_h=ckpt['hidden_size'], proj_size=ckpt['r_proj_size'], ncoef=ckpt['ncoef'], ndiscriminators=ckpt['ndiscriminators'])


	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	model.eval()
	if args.cuda:
		model = model.to(device)

	enroll_data = None

	files_list = glob.glob(args.enroll_data+'*.scp')

	for file_ in files_list:
		if enroll_data is None:
			enroll_data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				enroll_data[k] = v

	files_list = glob.glob(args.test_data+'*.scp')

	test_data = None

	for file_ in files_list:
		if test_data is None:
			test_data = { k:v for k,v in read_mat_scp(file_) }
		else:
			for k,v in read_mat_scp(file_):
				test_data[k] = v

	unlab_emb = None

	if args.unlab_data:

		files_list = glob.glob(args.unlab_data+'*.scp')

		unlab_emb = []

		for file_ in files_list:

			for k,v in read_mat_scp(file_):

				unlab_utt_data = prep_feats(v)

				if args.cuda:
					unlab_utt_data = unlab_utt_data.to(device)

				unlab_emb.append(model.forward(unlab_utt_data).detach().unsqueeze(0))


		unlab_emb=torch.cat(unlab_emb, 0).mean(0, keepdim=True)


	spk2utt = read_spk2utt(args.spk2utt)

	speakers_enroll, utterances_test, labels = read_trials(args.trials_path)

	print('\nAll data ready. Start of scoring')

	cos_scores = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	e2e_scores = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	fus_scores = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	out_e2e = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	out_cos = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	out_fus = {'emb_avg':[], 'scores_avg':[], 'scores_std':[], 'scores_median':[], 'scores_max':[], 'scores_min':[], 'scores_maxmin':[] }
	mem_embeddings_enroll_spk = {}
	mem_embeddings_enroll_spk_avg = {}
	mem_embeddings_test = {}

	with torch.no_grad():

		for i in range(len(labels)):

			## Get test embedding

			test_utt = utterances_test[i]

			try:

				emb_test = mem_embeddings_test[test_utt]

			except KeyError:

				test_utt_data = prep_feats(test_data[test_utt])

				if args.cuda:
					test_utt_data = test_utt_data.to(device)

				emb_test = model.forward(test_utt_data)[1].detach() if args.inner else model.forward(test_utt_data)[0].detach()

				if unlab_emb is not None:
					emb_test -= unlab_emb

				mem_embeddings_test[test_utt] = emb_test


			## Get enroll embeddings - both avg embedding (class prototype) and a list of all enroll embeddings are stored into dicts

			try:

				emb_avg_spk_enroll = mem_embeddings_enroll_spk_avg[speakers_enroll[i]]
				emb_spk_enroll = mem_embeddings_enroll_spk[speakers_enroll[i]]

			except KeyError:

				enroll_utts = list(np.random.choice(spk2utt[speakers_enroll[i]], min(len(spk2utt[speakers_enroll[i]]), args.max_nscores), replace=False))

				for enroll_utt in enroll_utts:

					emb_spk_enroll = []

					enroll_utt_data = prep_feats(enroll_data[enroll_utt])

					if args.cuda:
						enroll_utt_data = enroll_utt_data.to(device)

					emb_enroll = model.forward(enroll_utt_data)[1].detach() if args.inner else model.forward(enroll_utt_data)[0].detach()

					if unlab_emb is not None:
						emb_enroll -= unlab_emb

					emb_spk_enroll.append(emb_enroll)

				mem_embeddings_enroll_spk[speakers_enroll[i]] = emb_spk_enroll

				emb_avg_spk_enroll = torch.cat(emb_spk_enroll, 0).mean(0, keepdim=True)

				mem_embeddings_enroll_spk_avg[speakers_enroll[i]] = emb_avg_spk_enroll

			## get score based on class prototype

			pred = model.forward_bin(torch.cat([emb_avg_spk_enroll, emb_test],1))

			if model.ndiscriminators>1:
				e2e_scores['emb_avg'].append( torch.cat(pred, 1).mean(1).squeeze().item() )
			else:
				e2e_scores['emb_avg'].append( pred.squeeze().item() )

			cos_scores['emb_avg'].append( 0.5*(torch.nn.functional.cosine_similarity(emb_avg_spk_enroll, emb_test).mean().item()+1.) )
			fus_scores['emb_avg'].append( (e2e_scores['emb_avg'][-1]+cos_scores['emb_avg'][-1])*0.5 )

			## get score based on class prototype

			raw_scores_e2e = []
			raw_scores_cos = []
			raw_scores_fus = []

			for emb_enroll in emb_spk_enroll:

				pred = model.forward_bin(torch.cat([emb_enroll, emb_test],1))

				if model.ndiscriminators>1:
					raw_scores_e2e.append( torch.cat(pred, 1).mean(1).squeeze().item() )
				else:
					raw_scores_e2e.append( pred.squeeze().item() )

				raw_scores_cos.append( 0.5*(torch.nn.functional.cosine_similarity(emb_enroll, emb_test).mean().item()+1.) )
				raw_scores_fus.append( (raw_scores_e2e[-1]+raw_scores_cos[-1])*0.5 )


			e2e_scores['scores_avg'].append(np.mean(raw_scores_e2e))
			e2e_scores['scores_std'].append(np.std(raw_scores_e2e))
			e2e_scores['scores_median'].append(np.median(raw_scores_e2e))
			e2e_scores['scores_max'].append(np.max(raw_scores_e2e))
			e2e_scores['scores_min'].append(np.min(raw_scores_e2e))
			e2e_scores['scores_maxmin'].append(np.max(raw_scores_e2e)-np.min(raw_scores_e2e))

			cos_scores['scores_avg'].append(np.mean(raw_scores_cos))
			cos_scores['scores_std'].append(np.std(raw_scores_cos))
			cos_scores['scores_median'].append(np.median(raw_scores_cos))
			cos_scores['scores_max'].append(np.max(raw_scores_cos))
			cos_scores['scores_min'].append(np.min(raw_scores_cos))
			cos_scores['scores_maxmin'].append(np.max(raw_scores_cos)-np.min(raw_scores_cos))

			fus_scores['scores_avg'].append(np.mean(raw_scores_fus))
			fus_scores['scores_std'].append(np.std(raw_scores_fus))
			fus_scores['scores_median'].append(np.median(raw_scores_fus))
			fus_scores['scores_max'].append(np.max(raw_scores_fus))
			fus_scores['scores_min'].append(np.min(raw_scores_fus))
			fus_scores['scores_maxmin'].append(np.max(raw_scores_fus)-np.min(raw_scores_fus))

			for score_type in cos_scores:
				out_e2e[score_type].append([enroll_utt, test_utt, e2e_scores[score_type][-1]])
				out_cos[score_type].append([enroll_utt, test_utt, cos_scores[score_type][-1]])
				out_fus[score_type].append([enroll_utt, test_utt, fus_scores[score_type][-1]])

	print('\nScoring done')

	for score_type in cos_scores:

		with open(args.out_path+'e2e_'+score_type+'_spk.out', 'w') as f:
			for el in out_e2e:
				item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
				f.write("%s" % item)

		with open(args.out_path+'cos_'+score_type+'_spk.out', 'w') as f:
			for el in out_cos:
				item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
				f.write("%s" % item)

		with open(args.out_path+'fus_'+score_type+'_spk.out', 'w') as f:
			for el in out_fus:
				item = el[0] + ' ' + el[1] + ' ' + str(el[2]) + '\n'
				f.write("%s" % item)

	for score_type in cos_scores:

		print('\nResults for scores of type: []'.format(score_type))

		e2e_scores = np.asarray(e2e_scores[score_type])
		cos_scores = np.asarray(cos_scores[score_type])
		fus_scores = np.asarray(fus_scores[score_type])
		labels = np.asarray(labels)

		eer, auc, avg_precision, acc, threshold = compute_metrics(labels, e2e_scores)
		print('\nE2E eval:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

		eer, auc, avg_precision, acc, threshold = compute_metrics(labels, cos_scores)
		print('\nCOS eval:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}'.format(eer, auc, avg_precision, acc, threshold))

		eer, auc, avg_precision, acc, threshold = compute_metrics(labels, fus_scores)
		print('\nFUS eval:')
		print('ERR, AUC,  Average Precision, Accuracy and corresponding threshold: {}, {}, {}, {}, {}\n'.format(eer, auc, avg_precision, acc, threshold))
