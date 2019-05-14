import argparse
import h5py
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

def read_spk2utt(path):
	with open(path, 'r') as file:
		rows = file.readlines()

	spk2utt_dict = {}

	for row in rows:
		spk_utts = row.replace('\n','').split(' ')
		spk2utt_dict[spk_utts[0]] = spk_utts[1:]

	return spk2utt_dict

def parse_spk2utt(spk2utt_dict, min_utt=1, plot=False):

	spk2count_list = []

	for spk, utt_list in spk2utt_dict.items():
		spk2count_list.append(len(utt_list))

	print('Original spk2utt:')

	print('Number of speakers: {}'.format(len(spk2count_list)))
	print('Number of recordings: {}'.format(np.sum(spk2count_list)))
	print('Max: {}, Min: {}, AVG: {}, STD: {} recordings per speaker'.format(np.max(spk2count_list), np.min(spk2count_list), np.mean(spk2count_list), np.std(spk2count_list)))

	print('Filtered spk2utt:')

	spk2count_min = [i for i in spk2count_list if i>= min_utt]

	print('Number of speakers: {}'.format(len(spk2count_min)))
	print('Number of recordings: {}'.format(np.sum(spk2count_min)))
	print('Max: {}, Min: {}, AVG: {}, STD: {} recordings per speaker'.format(np.max(spk2count_min), np.min(spk2count_min), np.mean(spk2count_min), np.std(spk2count_min)))

	if plot:
		plt.figure(1)
		plt.hist(spk2count_list, bins=80)
		plt.show()

def hdf_to_spk2utt(hdf_path):

	open_file = h5py.File(hdf_path, 'r')

	speakers_list = list(open_file)

	spk2utt_ = {}

	for spk in speakers_list:
		spk2utt_[spk] = list(open_file[spk])

	open_file.close()

	return spk2utt_

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--min-num-utt', type=int, default=5)
	parser.add_argument('--path-to-spk2utt', type=str, default=None)
	parser.add_argument('--path-to-hdf', type=str, default=None)
	parser.add_argument('--plot', action='store_true', default=False)
	args = parser.parse_args()

	if args.path_to_spk2utt:
		spk2utt = read_spk2utt(args.path_to_spk2utt)
		print('spk2utt:')
		print('')
		parse_spk2utt(spk2utt_dict=spk2utt, min_utt=args.min_num_utt, plot=args.plot)

	if args.path_to_hdf:
		print('')
		print('hdf:')
		print('')
		spk2utt_hdf = hdf_to_spk2utt(hdf_path=args.path_to_hdf)
		parse_spk2utt(spk2utt_dict=spk2utt_hdf, min_utt=args.min_num_utt)
