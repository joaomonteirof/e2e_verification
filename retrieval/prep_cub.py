import argparse
import numpy as np
import os
import pathlib
import shutil

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CUB data preparation')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to scp files with features')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	args = parser.parse_args()

	pathlib.Path(os.path.join(args.out_path, 'train')).mkdir(parents=True, exist_ok=True)
	pathlib.Path(os.path.join(args.out_path, 'test')).mkdir(parents=True, exist_ok=True)

	train_list, test_list = list(range(1,101)), list(range(101,201))
	train_folders, test_folders = [], []

	print('Start of data preparation')

	folders_list = [x[0] for x in os.walk(args.path_to_data)]

	for folder in folders_list:
		try:
			class_id = int(folder.split('/')[-1].split('.')[0])
		except ValueError:
			continue

		if class_id in train_list:
			train_folders.append(folder)
		elif class_id in test_list:
			test_folders.append(folder)

	for folder in train_folders:
		source_path = folder
		tgt_path = os.path.join(args.out_path, 'train', folder.split('/')[-1])
		shutil.move(source_path, tgt_path)

	for folder in test_folders:
		source_path = folder
		tgt_path = os.path.join(args.out_path, 'test', folder.split('/')[-1])
		shutil.move(source_path, tgt_path)