import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import subprocess
import shlex
from utils import strided_app

class Loader(Dataset):

	def __init__(self, hdf5_name, transformation):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.transformation = transformation

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		example_1, example_2, example_3, example_4, example_5, clss, y = self.example_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		example_1_data = self.transformation( torch.from_numpy(self.open_file[clss][example_1][:,...]) )
		example_2_data = self.transformation( torch.from_numpy(self.open_file[clss][example_2][:,...]) )
		example_3_data = self.transformation( torch.from_numpy(self.open_file[clss][example_3][:,...]) )
		example_4_data = self.transformation( torch.from_numpy(self.open_file[clss][example_4][:,...]) )
		example_5_data = self.transformation( torch.from_numpy(self.open_file[clss][example_5][:,...]) )

		return example_1_data.contiguous(), example_2_data.contiguous(), example_3_data.contiguous(), example_4_data.contiguous(), example_5_data.contiguous(), y

	def __len__(self):
		return len(self.example_list)

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.class2file = {}
		self.clss2label = {}

		for i, clss in enumerate(open_file):
			clss_example_list = list(open_file[clss])
			self.class2file[clss] = clss_example_list
			self.clss2label[clss] = torch.LongTensor([i])

		open_file.close()

		self.n_classes = len(self.class2file)

	def update_lists(self):

		self.example_list = []

		for i, clss in enumerate(self.class2file):
			clss_file_list = np.random.permutation(self.class2file[clss])

			idxs = strided_app(np.arange(len(clss_file_list)), 5, 5)

			for idxs_list in idxs:
				if len(idxs_list)==5:
					self.example_list.append([clss_file_list[file_idx] for file_idx in idxs_list])
					self.example_list[-1].append(clss)
					self.example_list[-1].append(self.clss2label[clss])

if __name__=='__main__':

	import torch.utils.data
	from torchvision import transforms
	import argparse

	parser = argparse.ArgumentParser(description='Test data loader')
	parser.add_argument('--hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
	args = parser.parse_args()

	transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(84, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])	
	dataset = Loader(args.hdf_file, transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

	loader.dataset.update_lists()

	print('Dataset length: {}, {}'.format(len(loader.dataset), len(loader.dataset.example_list)))

	for batch in loader:
		utt_1, utt_2, utt_3, utt_4, utt_5, y = batch

	print(utt_1.size(), utt_2.size(), utt_3.size(), utt_4.size(), utt_5.size(), y.size())

	print(y)