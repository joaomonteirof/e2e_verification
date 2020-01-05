import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn import metrics

import torch
import itertools
import os
import sys
import pickle
from time import sleep

def parse_args_for_log(args):
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		if args_dict[arg_key] is None:
			args_dict[arg_key] = 'None'

	return args_dict

def get_classifier_config_from_cp(ckpt):
	keys=ckpt['model_state'].keys()
	classifier_params=[]
	out_proj_params=[]
	for x in keys:
		if 'classifier' in x:
			classifier_params.append(x)
		elif 'out_proj' in x:
			out_proj_params.append(x)
	return max(len(classifier_params)//2 - 1, 1), ckpt['model_state']['classifier.0.weight'].size(0), 'am_softmax' if len(out_proj_params)==1 else 'softmax'

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

def parse_spk2utt(spk2utt):

	utt_list, spk_list = [], []

	for spk, utts in spk2utt.items():
		utt_list.extend(utts)
		spk_list.extend([spk]*len(utts))

	return utt_list, spk_list

def create_trials(spk2utt):

	utt_list, spk_list = parse_spk2utt(spk2utt)

	enroll_utts, test_utts, labels = [], [], []

	for prod_exs in itertools.combinations(list(range(len(spk_list))), 2):

		enroll_utts.append(utt_list[prod_exs[0]])
		test_utts.append(utt_list[prod_exs[1]])

		if spk_list[prod_exs[0]]==spk_list[prod_exs[1]]:
			labels.append(1)
		else:
			labels.append(0)

	return enroll_utts, test_utts, labels

def calibrate(scores):

	max_ = np.max(scores)
	min_ = np.min(scores)

	return (scores-min_)/(max_-min_)

def compute_MAD(scores_set, x_median):
	return np.median(np.abs(scores_set - x_median))

def is_outlier(x, x_median, MAD):
	M = np.abs(.6745*(x - x_median)/MAD)
	if M>3.5:
		return True
	else:
		return False

def get_non_outliers(scores_set):
	non_outliers = []
	median = np.median(scores_set)
	MAD = compute_MAD(scores_set, median)
	for score in scores_set:
		if not is_outlier(score, median, MAD):
			non_outliers.append(score)

	return non_outliers

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0])

def get_freer_gpu(trials=10):
	sleep(5)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr

	t = np.nanargmin(np.abs(fnr-fpr))
	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	return eer

def compute_metrics(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	t = np.nanargmin(np.abs(fnr-fpr))

	eer_threshold = thresholds[t]

	eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
	eer = (eer_low+eer_high)*0.5

	auc = metrics.auc(fpr, tpr)

	avg_precision = metrics.average_precision_score(y, y_score)

	pred = np.asarray([1 if score > eer_threshold else 0 for score in y_score])
	acc = metrics.accuracy_score(y ,pred)

	return eer, auc, avg_precision, acc, eer_threshold

def read_trials(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	enroll_utt_list, test_utt_list, labels_list = [], [], []

	for line in utt_labels:
		enroll_utt, test_utt, label = line.split(' ')
		enroll_utt_list.append(enroll_utt)
		test_utt_list.append(test_utt)
		labels_list.append(1 if (label=='target\n' or label=='tgt\n') else 0)

	return enroll_utt_list, test_utt_list, labels_list

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

def read_utt2spk(path):
	with open(path, 'r') as file:
		pairs = file.readlines()

	utt2spk_dict = {}

	for pair in pairs:
		utt, spk = pair.split(' ')
		utt2spk_dict[utt] = spk.replace('\n','')

	return utt2spk_dict
