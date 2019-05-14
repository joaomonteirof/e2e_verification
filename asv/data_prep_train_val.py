import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

def read_utt2spk(path):
	with open(path, 'r') as file:
		pairs = file.readlines()

	utt2spk_dict = {}

	for pair in pairs:
		utt, spk = pair.split(' ')
		utt2spk_dict[utt] = spk.replace('\n','')

	return utt2spk_dict

def read_spk2utt(path, min_recs):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		if len(spk_utts[1:])>=min_recs:
			spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to scp files with features')
	parser.add_argument('--data-info-path', type=str, default='./data/', metavar='Path', help='Path to spk2utt and utt2spk')
	parser.add_argument('--spk2utt', type=str, default=None, metavar='Path', help='Path to spk2utt')
	parser.add_argument('--utt2spk', type=str, default=None, metavar='Path', help='Path to utt2spk')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--n-val-speakers', type=int, default=10, help='Number of speakers for valid data')
	parser.add_argument('--min-recordings', type=int, default=-1, help='Minimum number of recordings per speaker')
	args = parser.parse_args()

	if os.path.isfile(args.out_path+'train_'+args.out_name):
		os.remove(args.out_path+'train_'+args.out_name)
		print(args.out_path+'train_'+args.out_name+' Removed')

	if os.path.isfile(args.out_path+'valid_'+args.out_name):
		os.remove(args.out_path+'valid_'+args.out_name)
		print(args.out_path+'valid_'+args.out_name+' Removed')

	utt2spk = read_utt2spk(args.utt2spk if args.utt2spk else args.data_info_path+'utt2spk')
	spk2utt = read_spk2utt(args.spk2utt if args.spk2utt else args.data_info_path+'spk2utt', args.min_recordings)

	speakers_list = list(spk2utt.keys())

	val_idxs = np.random.choice(np.arange(len(speakers_list)), replace=False, size=args.n_val_speakers)
	val_spk_list = [speakers_list[i] for i in val_idxs]
	train_spk_list = [spk_ for spk_ in speakers_list if spk_ not in val_spk_list]

	scp_list = glob.glob(args.path_to_data + '*.scp')

	if len(scp_list)<1:
		print('Nothing found at {}.'.format(args.path_to_data))
		exit(1)

	print('Start of data preparation')

	train_hdf = h5py.File(args.out_path+'train_'+args.out_name, 'a')
	valid_hdf = h5py.File(args.out_path+'valid_'+args.out_name, 'a')

	for spk in train_spk_list:
		train_hdf.create_group(spk)

	for spk in val_spk_list:
		valid_hdf.create_group(spk)

	for file_ in scp_list:

		print('Processing file {}'.format(file_))

		data = { k:m for k,m in read_mat_scp(file_) }

		for i, utt in enumerate(data):

			speaker = utt2spk[utt]

			if speaker in spk2utt:
				if speaker in val_spk_list:
					hdf = valid_hdf
				else:
					hdf = train_hdf
			else:
				continue

			data_ = data[utt]
			#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
			features = data_.T

			if features.shape[0]>0:
				features = np.expand_dims(features, 0)
				hdf[speaker].create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
			else:
				print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))

	hdf.close()
