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

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		index = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker_1 = self.speakers_list[index]
		spk1_utt_list = self.spk2utt[speaker_1]
		idx1, idx2 = np.random.choice(np.arange(len(spk1_utt_list)), replace=True, size=2)

		utt_1 = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx1]] )
		utt_1 = torch.from_numpy( utt_1 )

		utt_p = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx2]] )
		utt_p = torch.from_numpy( utt_p )

		neg_speaker_idx = index

		while neg_speaker_idx == index:
			neg_speaker_idx = np.random.randint(len(self.speakers_list))

		neg_speaker = self.speakers_list[neg_speaker_idx]
		nspk_utt_list = self.spk2utt[neg_speaker]

		n_idx = np.random.randint(len(nspk_utt_list))
		utt_n = self.prep_utterance( self.open_file[neg_speaker][nspk_utt_list[n_idx]] )
		utt_n = torch.from_numpy( utt_n ).float().contiguous()

		return utt_1, utt_p, utt_n

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

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_softmax(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, n_cycles=100):
		super(Loader_softmax, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles
		self.max_nb_frames = int(max_nb_frames)

		self.create_lists()

		self.open_file = None

	def __getitem__(self, index):

		index = index % len(self.speakers_list)

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		speaker_1 = self.speakers_list[index]
		spk1_utt_list = self.spk2utt[speaker_1]
		idx1, idx2 = np.random.choice(np.arange(len(spk1_utt_list)), replace=True, size=2)

		utt_1 = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx1]] )
		utt_1 = torch.from_numpy( utt_1 )

		utt_p = self.prep_utterance( self.open_file[speaker_1][spk1_utt_list[idx2]] )
		utt_p = torch.from_numpy( utt_p )

		neg_speaker_idx = index

		while neg_speaker_idx == index:
			neg_speaker_idx = np.random.randint(len(self.speakers_list))

		neg_speaker = self.speakers_list[neg_speaker_idx]
		nspk_utt_list = self.spk2utt[neg_speaker]

		n_idx = np.random.randint(len(nspk_utt_list))
		utt_n = self.prep_utterance( self.open_file[neg_speaker][nspk_utt_list[n_idx]] )
		utt_n = torch.from_numpy( utt_n ).float().contiguous()

		return utt_1, utt_p, utt_n, torch.LongTensor([index])

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

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

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

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()

class Loader_test(Dataset):

	def __init__(self, hdf5_name, max_nb_frames, examples_per_speaker=5):
		super(Loader_test, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)
		self.examples_per_speaker = int(examples_per_speaker)
		self.last_index = 0

		self.create_lists()
		self.set_maxlen_spklist()
		self.set_indices()

		self.open_file = None
		self.n_speakers = len(self.speakers_list)

	def __getitem__(self, index):

		speaker_idx = index % self.n_speakers
		second_ind = index // self.n_speakers

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		if index < self.last_index:
			self.set_indices()

		speaker = self.speakers_list[speaker_idx]
		utt_list = self.spk2utt[speaker]

		idxs, utterances = self.indices[speaker][second_ind], []

		for i in idxs:
			utt = self.prep_utterance( self.open_file[speaker][utt_list[i]] )
			utterances.append( torch.from_numpy( utt ).float().contiguous() )

		self.last_index = index

		return torch.cat(utterances, 0).unsqueeze(1), torch.LongTensor(self.examples_per_speaker*[speaker_idx])

	def __len__(self):
		return len(self.speakers_list)*len(self.indices[self.speakers_list[0]])

	def set_indices(self):
		indices = {}

		for spk in self.spk2utt:
			spk_len = len(self.spk2utt[spk])

			if spk_len >= self.maxlen:
				spk_ind = np.random.choice(np.arange(spk_len), size=self.maxlen, replace=False)
			else:
				spk_ind = np.concatenate([np.random.permutation(np.arange(spk_len)), np.random.randint(0, spk_len, self.maxlen-spk_len)], 0)

			indices[spk] = strided_app(spk_ind, self.examples_per_speaker, self.examples_per_speaker)

		self.indices = indices

	def set_maxlen_spklist(self):
		open_file = h5py.File(self.hdf5_name, 'r')
		spk_list_ascii = open_file['spk_list']
		self.speakers_list = [spk.decode('utf-8') for spk in spk_list_ascii]

		length_list = []

		for spk in self.spk2utt:
			length_list.append(len(self.spk2utt[spk]))

		self.maxlen = int(np.mean(length_list))

		open_file.close()

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

		self.spk2utt = {}

		for spk in self.speakers_list:

			self.spk2utt[spk] = list(open_file[spk])

		open_file.close()
