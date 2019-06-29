import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import subprocess
import shlex
from numpy.lib.stride_tricks import as_strided

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

class Loader(Dataset):

	def __init__(self, hdf5_name, max_nb_frames):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		utt = self.utt_list[index]
		spk = self.utt2spk[utt]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = self.prep_utterance( self.open_file[spk][utt] )
		utt_data = torch.from_numpy( utt_data )

		utt_1, utt_2, utt_3, utt_4 = np.random.choice(self.spk2utt[spk], 4)

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )

		return utt_data, utt_1_data, utt_2_data, utt_3_data, utt_4_data, self.utt2label[utt]

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		if data.shape[2]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[2]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[0]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		return data_

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.n_speakers = len(open_file)

		self.utt2label = {}
		self.utt2spk = {}
		self.spk2utt = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			for utt in spk_utt_list:
				self.utt2label[utt] = torch.LongTensor([i])
				self.utt2spk[utt] = spk
				self.utt_list.append(utt)

		open_file.close()
