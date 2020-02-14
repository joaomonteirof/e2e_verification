import argparse
import numpy as np
import os
from tqdm import tqdm
import scipy.io as sio
import pathlib
import shutil

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Cars data preparation')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to scp files with features')
	parser.add_argument('--path-to-metadata', type=str, default='./data/metada.mat', metavar='Path', help='Path to metadata')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	args = parser.parse_args()

	metadata = sio.loadmat(args.path_to_metadata)
	annotations = metadata['annotations']
	annotations_labels = annotations['class']
	unique_labels = np.sort(np.unique(annotations_labels))

	train_labels, test_labels = unique_labels[:len(unique_labels)//2], unique_labels[len(unique_labels)//2:]

	print('Start of data preparation')

	classes = metadata['annotations']['class'][0]
	paths = metadata['annotations']['relative_im_path'][0]

	data_iterator = tqdm(enumerate(classes), total=len(classes))

	for i, class_id in data_iterator:

		if class_id in train_labels:
			folder = 'train'
		elif class_id in test_labels:
			folder = 'test'
		else:
			print('\nClass id {} not found !!\n'.format(class_id))
			continue

		class_id = str(class_id[0][0])
		source_path = os.path.join(args.path_to_data, paths[i][0])
		tgt_path = os.path.join(args.out_path, folder, class_id, paths[i][0].split('/')[-1])

		pathlib.Path(os.path.join(args.path_to_data, folder, class_id)).mkdir(parents=True, exist_ok=True)

		shutil.move(source_path, tgt_path)