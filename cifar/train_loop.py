import torch
import torch.nn.functional as F

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

from harvester import HardestNegativeTripletSelector, AllTripletSelector

from sklearn import metrics

def compute_eer(y, y_score):
	fpr, tpr, thresholds = metrics.roc_curve(y, y_score, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = thresholds[np.nanargmin(np.abs(fnr-fpr))]
	eer = fpr[np.nanargmin(np.abs(fnr-fpr))]

	return eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, patience, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, pretrain=False, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.pretrain = pretrain
		self.model = model
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = next(self.model.parameters()).device
		self.history = {'train_loss': [], 'train_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'bin_loss': [], 'bin_loss_batch': []}

		if self.valid_loader is not None:
			self.history['e2e_eer'] = []
			self.history['cos_eer'] = []
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True if self.verbose>0 else False, threshold=1e-4, min_lr=1e-7)
		else:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 100, 200, 300, 400], gamma=0.1)

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):

			np.random.seed()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			if self.pretrain:

				ce_epoch=0.0
				for t, batch in train_iter:
					ce = self.pretrain_step(batch)
					self.history['train_loss_batch'].append(ce)
					ce_epoch+=ce
					self.total_iters += 1

				self.history['train_loss'].append(ce_epoch/(t+1))

				if self.verbose>0:
					print('Train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			else:

				train_loss_epoch=0.0
				ce_loss_epoch=0.0
				bin_loss_epoch=0.0
				for t, batch in train_iter:
					train_loss, ce_loss, bin_loss = self.train_step(batch)
					self.history['train_loss_batch'].append(train_loss)
					self.history['ce_loss_batch'].append(ce_loss)
					self.history['bin_loss_batch'].append(bin_loss)
					train_loss_epoch+=train_loss
					ce_loss_epoch+=ce_loss
					bin_loss_epoch+=bin_loss
					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))
				self.history['ce_loss'].append(ce_loss_epoch/(t+1))
				self.history['bin_loss'].append(bin_loss_epoch/(t+1))

				if self.verbose>0:
					print(' ')
					print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
					print('CE loss: {:0.4f}'.format(self.history['ce_loss'][-1]))
					print('Binary classification loss: {:0.4f}'.format(self.history['bin_loss'][-1]))
					print(' ')

			if self.valid_loader is not None:

				e2e_scores, cos_scores, labels = None, None, None

				for t, batch in enumerate(self.valid_loader):
					e2e_scores_batch, cos_scores_batch, labels_batch = self.valid(batch)

					try:
						e2e_scores = np.concatenate([e2e_scores, e2e_scores_batch], 0)
						cos_scores = np.concatenate([cos_scores, cos_scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						e2e_scores, cos_scores, labels = e2e_scores_batch, cos_scores_batch, labels_batch

				self.history['e2e_eer'].append(compute_eer(labels, e2e_scores))
				self.history['cos_eer'].append(compute_eer(labels, cos_scores))

				if self.verbose>0:
					print(' ')
					print('Current e2e EER, best e2e EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['e2e_eer'][-1], np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))
					print('Current cos EER, best cos EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['cos_eer'][-1], np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))

				self.scheduler.step(np.min([self.history['e2e_eer'][-1], self.history['cos_eer'][-1]]))

			else:
				self.scheduler.step()

			if self.verbose>0:
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.valid_loader is not None and self.save_cp and (self.cur_epoch % save_every == 0 or self.history['e2e_eer'][-1] < np.min([np.inf]+self.history['e2e_eer'][:-1]) or self.history['cos_eer'][-1] < np.min([np.inf]+self.history['cos_eer'][:-1])):
					self.checkpointing()
			elif self.save_cp and self.cur_epoch % save_every == 0:
					self.checkpointing()

		if self.verbose>0:
			print('Training done!')

		if self.valid_loader is not None:
			if self.verbose>0:
				print('Best e2e eer and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))
				print('Best cos eer and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))

			return [np.min(self.history['e2e_eer']), np.min(self.history['cos_eer'])]
		else:
			return [np.min(self.history['train_loss'])]

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		x = x.to(self.device)
		y = y.to(self.device)

		embeddings = self.model.forward(x)

		embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

		ce_loss = F.cross_entropy(self.model.out_proj(embeddings_norm, y), y)

		# Get all triplets now for bin classifier
		triplets_idx = self.harvester.get_triplets(embeddings_norm.detach(), y)
		triplets_idx = triplets_idx.to(self.device)

		emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

		emb_ap = torch.cat([emb_a, emb_p],1)
		emb_an = torch.cat([emb_a, emb_n],1)
		emb_ = torch.cat([emb_ap, emb_an],0)

		y_ = torch.cat([torch.ones(emb_ap.size(0)), torch.zeros(emb_an.size(0))],0)
		y_ = y_.to(self.device)

		pred_bin = self.model.forward_bin(emb_).squeeze()

		loss_bin = torch.nn.BCEWithLogitsLoss()(pred_bin, y_)

		loss = ce_loss + loss_bin
		loss.backward()
		self.optimizer.step()

		return loss.item(), ce_loss.item(), loss_bin.item()


	def pretrain_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		x, y = x.to(self.device), y.to(self.device)

		embeddings = self.model.forward(utt)

		loss = F.cross_entropy(self.model.out_proj(embeddings), y.squeeze())

		loss.backward()
		self.optimizer.step()
		return loss.item()


	def valid(self, batch, n=10):

		self.model.eval()

		with torch.no_grad():

			x, y = batch

			x = x.to(self.device)
			y = y.to(self.device)

			embeddings = self.model.forward(x)

			embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

			ce_loss = F.cross_entropy(self.model.out_proj(embeddings_norm, y), y)

			# Get all triplets now for bin classifier
			triplets_idx = self.harvester.get_triplets(embeddings_norm.detach(), y)
			triplets_idx = triplets_idx.to(self.device)

			emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

			emb_ap = torch.cat([emb_a, emb_p],1)
			emb_an = torch.cat([emb_a, emb_n],1)

			e2e_scores_p = self.model.forward_bin(emb_ap).squeeze()
			e2e_scores_n = self.model.forward_bin(emb_an).squeeze()
			cos_scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			cos_scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

		return np.concatenate([e2e_scores_p.detach().cpu().numpy(), e2e_scores_n.detach().cpu().numpy()], 0), np.concatenate([cos_scores_p.detach().cpu().numpy(), cos_scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(e2e_scores_p.size(0)), np.zeros(e2e_scores_n.size(0))], 0)

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.scheduler.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		try:
			torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))
		except:
			torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))
