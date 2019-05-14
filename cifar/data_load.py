import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import as_strided

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

class Loader(Dataset):

	def __init__(self, hdf5_name, examples_per_class=5):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.examples_per_class = int(examples_per_class)

		self.set_maxlen_classlist()
		self.set_indices()

		self.open_file = None

		self.class2label = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

		self.n_classes = len(self.classes_list)

		self.last_index = 0

	def __getitem__(self, index):

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		if index < self.last_index:
			self.set_indices()

		class_idx = index % self.n_classes
		second_ind = index // self.n_classes

		class_ = self.classes_list[class_idx]

		idxs, samples = self.indices[class_][second_ind], []

		for idx in idxs:
			sample = self.open_file[class_][idx]
			samples.append( torch.from_numpy( sample ).unsqueeze(0).float().contiguous() )

		self.last_index = index

		return torch.cat(samples, 0), torch.LongTensor(5*[self.class2label[class_]])

	def __len__(self):
		return len(self.classes_list)*len(self.indices[self.classes_list[0]])

	def set_indices(self):
		indices = {}

		open_file = h5py.File(self.hdf5_name, 'r')

		for cls in open_file:
			cls_len = len(open_file[cls])

			if cls_len >= self.maxlen:
				cls_ind = np.random.choice(np.arange(cls_len), size=self.maxlen, replace=False)
			else:
				cls_ind = np.concatenate([np.random.permutation(np.arange(cls_len)), np.random.randint(0, cls_len, self.maxlen-cls_len)], 0)

			indices[cls] = strided_app(cls_ind, self.examples_per_class, self.examples_per_class)

		self.indices = indices
		open_file.close()

	def set_maxlen_classlist(self):
		open_file = h5py.File(self.hdf5_name, 'r')
		self.classes_list = list(open_file.keys())

		length_list = []

		for cls in open_file:
			length_list.append(len(open_file[cls]))

		self.maxlen = np.max(length_list)

		open_file.close()
