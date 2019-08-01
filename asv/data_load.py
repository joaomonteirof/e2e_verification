import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import subprocess
import shlex
from utils.utils import strided_app

class Loader(Dataset):

	def __init__(self, hdf5_name, max_nb_frames):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt_1, utt_2, utt_3, utt_4, utt_5, spk, y= self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_1_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_1] ) )
		utt_2_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_2] ) )
		utt_3_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_3] ) )
		utt_4_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_4] ) )
		utt_5_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt_5] ) )

		return utt_1_data.contiguous(), utt_2_data.contiguous(), utt_3_data.contiguous(), utt_4_data.contiguous(), utt_5_data, y

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

		self.spk2utt = {}
		self.spk2label = {}
		self.utt_list = []

		for i, spk in enumerate(open_file):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			self.spk2label[spk] = torch.LongTensor([i])

		open_file.close()

		self.n_speakers = len(self.spk2utt)

	def update_lists(self):

		self.utt_list = []

		for i, spk in enumerate(self.spk2utt):
			spk_utt_list = np.random.permutation(self.spk2utt[spk])

			idxs = strided_app(np.arange(len(spk_utt_list)),5,5)

			for idxs_list in idxs:
				if len(idxs_list)==5:
					self.utt_list.append([spk_utt_list[utt_idx] for utt_idx in idxs_list])
					self.utt_list[-1].append(spk)
					self.utt_list[-1].append(self.spk2label[spk])

class Loader_valid(Dataset):

	def __init__(self, hdf5_name, max_nb_frames):
		super(Loader_valid, self).__init__()
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

		return utt_data.contiguous(), utt_1_data.contiguous(), utt_2_data.contiguous(), utt_3_data.contiguous(), utt_4_data.contiguous(), self.utt2label[utt]

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
