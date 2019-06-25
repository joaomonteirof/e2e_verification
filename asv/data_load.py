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
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		utt = self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = self.prep_utterance( self.open_file[self.utt2spk[utt]][utt] )
		utt_data = torch.from_numpy( utt_data )

		return utt_data, self.utt2label[utt]

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
		self.utt_list = []

		for i, spk in self.speakers_list:
			for utt in list(open_file[spk]):
				self.utt2label[utt] = torch.LongTensor(i)
				self.utt2spk[utt] = spk
				self.utt_list.append(utt)

		open_file.close()

class Loader_mining(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100, examples_per_speaker=5):
		super(Loader_mining, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)
		self.examples_per_speaker = int(examples_per_speaker)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		speaker_idx = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utterances = []

		speaker = self.speakers_list[speaker_idx]
		utt_list = self.spk2utt[speaker]

		for i in range(self.examples_per_speaker):
			idx = np.random.randint(len(utt_list))
			utt = self.prep_utterance( self.open_file[speaker][utt_list[idx]] )
			utterances.append( torch.from_numpy( utt ).float().contiguous() )

		return torch.cat(utterances, 0).unsqueeze(1), torch.LongTensor(self.examples_per_speaker*[speaker_idx])

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

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

		self.speakers_list = list(open_file)
		self.n_speakers = len(open_file)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_pretrain(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100):
		super(Loader_pretrain, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		speaker_idx = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker = self.speakers_list[speaker_idx]
		utt_list = self.spk2utt[speaker]

		idx = np.random.randint(len(utt_list))
		utt = self.prep_utterance( self.open_file[speaker][utt_list[idx]] )
		utt = torch.from_numpy(utt).float().contiguous()

		return utt, torch.LongTensor([speaker_idx])

	def __len__(self):
		return len(self.speakers_list)*self.n_cycles

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

		self.speakers_list = list(open_file)
		self.n_speakers = len(open_file)

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()
