import torch
from torch.autograd import Variable
import torch.nn.init as init
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

	def __init__(self, model, optimizer, train_loader, valid_loader, margin, lambda_, patience, verbose=-1, cp_name=None, save_cp=False, checkpoint_path=None, checkpoint_epoch=None, swap=False, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, cp_name) if cp_name else os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True if verbose>0 else False, threshold=1e-4, min_lr=1e-8)
		self.total_iters = 0
		self.cur_epoch = 0
		self.lambda_ = lambda_
		self.swap = swap
		self.margin = margin
		self.harvester = HardestNegativeTripletSelector(margin=0.1, cpu=not self.cuda_mode)
		self.harvester_bin = AllTripletSelector()
		self.verbose = verbose
		self.save_cp = save_cp
		self.history = {'train_loss': [], 'train_loss_batch': [], 'triplet_loss': [], 'triplet_loss_batch': [], 'ce_loss': [], 'ce_loss_batch': [], 'bin_loss': [], 'bin_loss_batch': [], 'reg_entropy': [], 'reg_entropy_batch': [], 'e2e_eer':[], 'cos_eer':[]}

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while self.cur_epoch < n_epochs:

			np.random.seed()

			if self.verbose>0:
				print(' ')
				print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
				train_iter = tqdm(enumerate(self.train_loader))
			else:
				train_iter = enumerate(self.train_loader)

			train_loss_epoch=0.0
			triplet_loss_epoch=0.0
			ce_loss_epoch=0.0
			bin_loss_epoch=0.0
			entropy_reg_epoch=0.0
			for t, batch in train_iter:
				train_loss, triplet_loss, ce_loss, bin_loss, entropy_reg = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				self.history['triplet_loss_batch'].append(triplet_loss)
				self.history['ce_loss_batch'].append(ce_loss)
				self.history['bin_loss_batch'].append(bin_loss)
				self.history['reg_entropy_batch'].append(entropy_reg)
				train_loss_epoch+=train_loss
				triplet_loss_epoch+=triplet_loss
				ce_loss_epoch+=ce_loss
				bin_loss_epoch+=bin_loss
				entropy_reg_epoch+=entropy_reg
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))
			self.history['triplet_loss'].append(triplet_loss_epoch/(t+1))
			self.history['ce_loss'].append(ce_loss_epoch/(t+1))
			self.history['bin_loss'].append(bin_loss_epoch/(t+1))
			self.history['reg_entropy'].append(entropy_reg_epoch/(t+1))

			if self.verbose>0:
				print(' ')
				print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))
				print('CE loss: {:0.4f}'.format(self.history['ce_loss'][-1]))
				print('triplet loss: {:0.4f}'.format(self.history['triplet_loss'][-1]))
				print('Binary classification loss: {:0.4f}'.format(self.history['bin_loss'][-1]))
				print('Max entropy regularizer: {:0.4f}'.format(self.history['reg_entropy'][-1]))
				print(' ')

			# Validation

			tot_correct = 0
			tot_ = 0
			e2e_scores, cos_scores, labels = None, None, None

			for t, batch in enumerate(self.valid_loader):

				correct, total, e2e_scores_batch, cos_scores_batch, labels_batch = self.valid(batch)

				try:
					e2e_scores = np.concatenate([e2e_scores, e2e_scores_batch], 0)
					cos_scores = np.concatenate([cos_scores, cos_scores_batch], 0)
					labels = np.concatenate([labels, labels_batch], 0)
				except:
					e2e_scores, cos_scores, labels = e2e_scores_batch, cos_scores_batch, labels_batch

				tot_correct += correct
				tot_ += total

			self.history['e2e_eer'].append(compute_eer(labels, e2e_scores))
			self.history['cos_eer'].append(compute_eer(labels, cos_scores))
			self.history['ErrorRate'].append(1.-float(tot_correct)/tot_)

			if self.verbose>0:
				print(' ')
				print('Current, best validation error rate, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['ErrorRate'][-1], np.min(self.history['ErrorRate']), 1+np.argmin(self.history['ErrorRate'])))

				print(' ')
				print('Current, best validation e2e EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['e2e_eer'][-1], np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))

				print(' ')
				print('Current, best validation cos EER, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['cos_eer'][-1], np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))

			self.scheduler.step(np.min([self.history['e2e_eer'][-1], self.history['cos_eer'][-1]]))

			if self.verbose>0:
				print(' ')
				print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			if self.save_cp and (self.cur_epoch % save_every == 0 or (self.history['ErrorRate'][-1] < np.min([np.inf]+self.history['ErrorRate'][:-1])) or (self.history['e2e_eer'][-1] < np.min([np.inf]+self.history['e2e_eer'][:-1])) or (self.history['cos_eer'][-1] < np.min([np.inf]+self.history['cos_eer'][:-1]))):
				self.checkpointing()

			self.cur_epoch += 1

		if self.verbose>0:
			print('Training done!')

			if self.valid_loader is not None:
				print('Best error rate and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['ErrorRate']), 1+np.argmin(self.history['ErrorRate'])))
				print('Best e2e EER and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['e2e_eer']), 1+np.argmin(self.history['e2e_eer'])))
				print('Best cos EER and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['cos_eer']), 1+np.argmin(self.history['cos_eer'])))

		return np.min(self.history['e2e_eer']), np.min(self.history['cos_eer']), np.min(self.history['ErrorRate'])

	def train_step(self, batch):

		self.model.train()

		self.optimizer.zero_grad()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		out, embeddings = self.model.forward(x)

		embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

		loss_class = torch.nn.CrossEntropyLoss()(out, y)

		triplets_idx, entropy_indices = self.harvester.get_triplets(embeddings_norm.detach(), y)

		if self.cuda_mode:
			triplets_idx = triplets_idx.cuda()

		emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

		loss_metric = self.triplet_loss(emb_a, emb_p, emb_n)

		entropy_regularizer = -torch.log(torch.nn.functional.pairwise_distance(embeddings_norm, embeddings_norm[entropy_indices,:])).mean()*self.lambda_

		triplets_idx = self.harvester_bin.get_triplets(embeddings_norm.detach(), y)

		if self.cuda_mode:
			triplets_idx = triplets_idx.cuda(self.device)

		emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
		emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
		emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

		emb_ap = torch.cat([emb_a, emb_p],1)
		emb_an = torch.cat([emb_a, emb_n],1)
		emb_ = torch.cat([emb_ap, emb_an],0)

		y_ = torch.cat([torch.ones(emb_ap.size(0)), torch.zeros(emb_an.size(0))],0)

		if self.cuda_mode:
			y_ = y_.cuda(self.device)

		pred_bin = self.model.forward_bin(emb_).squeeze()

		loss_bin = torch.nn.BCEWithLogitsLoss()(pred_bin, y_)

		loss = loss_class + loss_metric + loss_bin + entropy_regularizer

		loss.backward()

		self.optimizer.step()

		return loss.item(), loss_class.item(), loss_metric.item(), loss_bin.item(), entropy_regularizer.item()

	def valid(self, batch):

		self.model.eval()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		with torch.no_grad():

			out, embeddings = self.model.forward(x)
			pred = F.softmax(out, dim=1).max(1)[1].long()
			correct = pred.squeeze().eq(y.squeeze()).detach().sum().item()

			triplets_idx = self.harvester_bin.get_triplets(embeddings, y)

			embeddings_norm = torch.div(embeddings, torch.norm(embeddings, 2, 1).unsqueeze(1).expand_as(embeddings))

			emb_a = torch.index_select(embeddings_norm, 0, triplets_idx[:, 0])
			emb_p = torch.index_select(embeddings_norm, 0, triplets_idx[:, 1])
			emb_n = torch.index_select(embeddings_norm, 0, triplets_idx[:, 2])

			emb_ap = torch.cat([emb_a, emb_p],1)
			emb_an = torch.cat([emb_a, emb_n],1)

			e2e_scores_p = self.model.forward_bin(emb_ap).squeeze()
			e2e_scores_n = self.model.forward_bin(emb_an).squeeze()
			cos_scores_p = torch.nn.functional.cosine_similarity(emb_a, emb_p)
			cos_scores_n = torch.nn.functional.cosine_similarity(emb_a, emb_n)

		return correct, x.size(0), np.concatenate([e2e_scores_p.detach().cpu().numpy(), e2e_scores_n.detach().cpu().numpy()], 0), np.concatenate([cos_scores_p.detach().cpu().numpy(), cos_scores_n.detach().cpu().numpy()], 0), np.concatenate([np.ones(e2e_scores_p.size(0)), np.zeros(e2e_scores_n.size(0))], 0)

	def triplet_loss(self, emba, embp, embn, reduce_=True):

		loss_ = torch.nn.TripletMarginLoss(margin=self.margin, p=2.0, eps=1e-06, swap=self.swap, reduction='mean' if reduce_ else 'none')(emba, embp, embn)

		return loss_

	def checkpointing(self):

		# Checkpointing
		if self.verbose>0:
			print(' ')
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

			ckpt = torch.load(ckpt)
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

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!')

	def initialize_params(self):
		for layer in self.model.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
