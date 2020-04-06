import torch
import torch.nn.functional as F

import numpy as np
import random

import os
from tqdm import tqdm

from harvester import HardestNegativeTripletSelector, AllTripletSelector
from models.losses import LabelSmoothingLoss
from utils import compute_eer, adjust_learning_rate
from data_load import Loader

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, max_gnorm, label_smoothing, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, pretrain=False, cuda=True, logger=None):
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
		self.max_gnorm = max_gnorm
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.harvester = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.device = next(self.model.parameters()).device
		self.logger = logger
		self.history = {'train_loss': [], 'train_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'bin_loss': [], 'bin_loss_batch': []}
		self.disc_label_smoothing = label_smoothing
		self.best_e2e_eer, self.best_cos_eer = np.inf, np.inf

		if label_smoothing>0.0:
			n_classes = self.train_loader.dataset.n_classes if isinstance(self.train_loader.dataset, Loader) else len(self.train_loader.dataset.classes)
			self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=n_classes)
		else:
			self.ce_criterion = torch.nn.CrossEntropyLoss()

		if self.valid_loader is not None:
			self.history['e2e_eer'] = []
			self.history['cos_eer'] = []

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1, eval_every=1000):

		while (self.cur_epoch < n_epochs):

			self.cur_epoch += 1

			np.random.seed()
			if isinstance(self.train_loader.dataset, Loader):
				self.train_loader.dataset.update_lists()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			if self.pretrain:
				self.save_epoch_cp = False
				ce_epoch=0.0
				for t, batch in train_iter:
					ce = self.pretrain_step(batch)
					self.history['train_loss_batch'].append(ce)
					ce_epoch+=ce
					if self.logger:
						self.logger.add_scalar('Train/Cross entropy', ce, self.total_iters)
						self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)

					self.total_iters += 1

				self.history['train_loss'].append(ce_epoch/(t+1))

				if self.verbose>0:
					print('Train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			else:
				self.save_epoch_cp = False
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
					if self.logger:
						self.logger.add_scalar('Train/Total train Loss', train_loss, self.total_iters)
						self.logger.add_scalar('Train/Binary class. Loss', bin_loss, self.total_iters)
						self.logger.add_scalar('Train/Cross enropy', ce_loss, self.total_iters)
						self.logger.add_scalar('Info/LR', self.optimizer.optimizer.param_groups[0]['lr'], self.total_iters)

					if eval_every % self.total_iters == 0:
						self.evaluate()
						if self.save_cp and ( self.history['e2e_eer'][-1] < np.min([np.inf]+self.history['e2e_eer'][:-1]) or self.history['cos_eer'][-1] < np.min([np.inf]+self.history['cos_eer'][:-1]) ):
								self.checkpointing()
								self.save_epoch_cp = True

					self.total_iters += 1

				self.history['train_loss'].append(train_loss_epoch/(t+1))
				self.history['ce_loss'].append(ce_loss_epoch/(t+1))
				self.history['bin_loss'].append(bin_loss_epoch/(t+1))

				if self.verbose>0:
					print(' ')
					print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
					print('CE loss: {:0.4f}'.format(self.history['ce_loss'][-1]))
					print('Binary classification loss: {:0.4f}'.format(self.history['bin_loss'][-1]))
					print('Current LR: {}'.format(self.optimizer.optimizer.param_groups[0]['lr']))
					print(' ')

			if self.save_cp and self.cur_epoch % save_every == 0 and not self.save_epoch_cp:
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

		if isinstance(self.train_loader.dataset, Loader):
			x_1, x_2, x_3, x_4, x_5, y = batch
			x = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=0)
			y = torch.cat(5*[y], dim=0).squeeze().contiguous()
		else:
			x, y = batch

		x = x.to(self.device, non_blocking=True)
		y = y.to(self.device, non_blocking=True)

		embeddings, out = self.model.forward(x)

		ce_loss = self.ce_criterion(self.model.out_proj(out, y), y)

		# Get all triplets now for bin classifier
		triplets_idx = self.harvester.get_triplets(embeddings.detach(), y)
		triplets_idx = triplets_idx.to(self.device, non_blocking=True)

		emb_a = torch.index_select(embeddings, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings, 0, triplets_idx[:, 2])

		if random.random() > 0.5:
			emb_a += torch.randn_like(emb_a)*random.choice([1e-2, 1e-3, 1e-4, 1e-5])
			emb_p += torch.randn_like(emb_p)*random.choice([1e-2, 1e-3, 1e-4, 1e-5])
			emb_n += torch.randn_like(emb_n)*random.choice([1e-2, 1e-3, 1e-4, 1e-5])

		emb_ap = torch.cat([emb_a, emb_p],1)
		emb_an = torch.cat([emb_a, emb_n],1)
		emb_ = torch.cat([emb_ap, emb_an],0)

		y_ = torch.cat([torch.rand(emb_ap.size(0))*self.disc_label_smoothing+(1.0-self.disc_label_smoothing), torch.rand(emb_an.size(0))*self.disc_label_smoothing],0) if isinstance(self.ce_criterion, LabelSmoothingLoss) else torch.cat([torch.ones(emb_ap.size(0)), torch.zeros(emb_an.size(0))],0)
		y_ = y_.to(self.device, non_blocking=True)

		pred_bin = self.model.forward_bin(emb_).squeeze()

		loss_bin = torch.nn.BCELoss()(pred_bin, y_)

		loss = ce_loss + loss_bin
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()

		if self.logger:
			self.logger.add_scalar('Info/Grad_norm', grad_norm, self.total_iters)

		return loss.item(), ce_loss.item(), loss_bin.item()


	def pretrain_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		x, y = batch

		x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).squeeze()

		embeddings, out = self.model.forward(utt)

		loss = F.cross_entropy(self.model.out_proj(out, y), y)

		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gnorm)
		self.optimizer.step()
		return loss.item()


	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			if isinstance(self.valid_loader.dataset, Loader):
				x_1, x_2, x_3, x_4, x_5, y = batch
				x = torch.cat([x_1, x_2, x_3, x_4, x_4], dim=0)
				y = torch.cat(5*[y], dim=0).squeeze().contiguous()
			else:
				x, y = batch

			x = x.to(self.device, non_blocking=True)
			y = y.to(self.device, non_blocking=True)

			embeddings, out = self.model.forward(x)

			# Get all triplets now for bin classifier
			triplets_idx = self.harvester.get_triplets(embeddings.detach(), y)
			triplets_idx = triplets_idx.to(self.device, non_blocking=True)

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

	def evaluate(self):

		if self.verbose>0:
			print('\nIteration - Epoch {} - {}'.format(self.total_iters, self.cur_epoch))

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

		if self.history['e2e_eer'][-1]<self.best_e2e_eer:
			self.best_e2e_eer = self.history['e2e_eer'][-1]
			self.best_e2e_eer_epoch = self.cur_epoch

		if self.history['cos_eer'][-1]<self.best_cos_eer:
			self.best_cos_eer = self.history['cos_eer'][-1]
			self.best_cos_eer_epoch = self.cur_epoch

		if self.logger:
			self.logger.add_scalar('Valid/E2E EER', self.history['e2e_eer'][-1], self.total_iters)
			self.logger.add_scalar('Valid/Best E2E EER', np.min(self.history['e2e_eer']), self.total_iters)
			self.logger.add_scalar('Valid/Cosine EER', self.history['cos_eer'][-1], self.total_iters)
			self.logger.add_scalar('Valid/Best Cosine EER', np.min(self.history['cos_eer']), self.total_iters)
			self.logger.add_pr_curve('E2E ROC', labels=labels, predictions=e2e_scores, global_step=self.total_iters)
			self.logger.add_pr_curve('Cosine ROC', labels=labels, predictions=cos_scores, global_step=self.total_iters)
			self.logger.add_histogram('Valid/COS_Scores', values=cos_scores, global_step=self.total_iters)
			self.logger.add_histogram('Valid/E2E_Scores', values=e2e_scores, global_step=self.total_iters)
			self.logger.add_histogram('Valid/Labels', values=labels, global_step=self.total_iters)

		if self.verbose>0:
			print(' ')
			print('Current e2e EER, best e2e EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['e2e_eer'][-1], np.min(self.history['e2e_eer']), self.best_e2e_eer))
			print('Current cos EER, best cos EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['cos_eer'][-1], np.min(self.history['cos_eer']), self.best_cos_eer))

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'dropout_prob': self.model.dropout_prob,
		'n_hidden': self.model.n_hidden,
		'hidden_size': self.model.hidden_size,
		'sm_type': self.model.sm_type,
		'n_classes': self.model.n_classes,
		'emb_size': self.model.emb_size,
		'optimizer_state': self.optimizer.state_dict(),
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
