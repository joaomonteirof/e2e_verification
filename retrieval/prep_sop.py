import argparse
import numpy as np
import os
from tqdm import tqdm
import scipy.io as sio
import pathlib
import shutil

def read_metadata(path):
	with open(path, 'r') as file:
		info = file.readlines()

	info = iter(info)
	next(info)

	label_list, path_list = [], []

	for line in info:
		_, label, _, im = line.split(' ')
		im = im.strip()
		label_list.append(label)
		path_list.append(im)

	return label_list, path_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Stanford online products data preparation')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to scp files with features')
	parser.add_argument('--path-to-metadata', type=str, default='./data/metada.mat', metavar='Path', help='Path to metadata')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	args = parser.parse_args()

	label_list, path_list = read_metadata(args.path_to_metadata)

	print('Start of data preparation')

	data_iterator = tqdm(enumerate(label_list), total=len(label_list))

	for i, class_id in data_iterator:

		source_path = os.path.join(args.path_to_data, path_list[i])
		tgt_path = os.path.join(args.out_path, class_id, path_list[i].split('/')[-1])

		pathlib.Path(os.path.join(args.out_path, class_id)).mkdir(parents=True, exist_ok=True)

		shutil.move(source_path, tgt_path)