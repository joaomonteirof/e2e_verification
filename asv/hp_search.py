from concurrent import futures
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader
import os
import sys

from utils.utils import *

def get_file_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	file_ = open(fname, 'wb')
	pickle.dump(None, file_)
	file_.close()

	return fname

# Training settings
parser=argparse.ArgumentParser(description='HP random search for ASV')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--model', choices=['resnet_stats', 'resnet_mfcc', 'resnet_lstm', 'resnet_small', 'resnet_large', 'TDNN', 'all'], default='resnet_lstm', help='Model arch according to input type')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()
args.cuda=True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, patience, latent_size, n_hidden, hidden_size, n_frames, model, ncoef, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, cp_path, softmax):

	if cuda:
		device=get_freer_gpu()

	train_dataset=Loader(hdf5_name=train_hdf_file, max_nb_frames=int(n_frames))
	train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

	valid_dataset = Loader(hdf5_name = valid_hdf_file, max_nb_frames = int(n_frames))
	valid_loader=torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

	if args.model == 'resnet_stats':
		model = model_.ResNet_stats(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet_mfcc':
		model = model_.ResNet_mfcc(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)
	if args.model == 'resnet_lstm':
		model = model_.ResNet_lstm(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet_small':
		model = model_.ResNet_small(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet_large':
		model = model_.ResNet_large(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'TDNN':
		model = model_.TDNN(n_z=int(latent_size), nh=int(n_hidden), n_h=int(hidden_size), proj_size=len(train_dataset.speakers_list), ncoef=ncoef, dropout_prob=dropout_prob, sm_type=softmax)

	if cuda:
		model=model.cuda(device)
	else:
		device=None

	optimizer=optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

	trainer=TrainLoop(model, optimizer, train_loader, valid_loader, patience=int(patience), verbose=-1, device=device, cp_name=get_file_name(cp_path), save_cp=False, checkpoint_path=cp_path, pretrain=False, cuda=cuda)

	return trainer.train(n_epochs=epochs)

lr=instru.var.OrderedDiscrete([0.1, 0.01, 0.001, 0.0001, 0.00001])
l2=instru.var.OrderedDiscrete([0.001, 0.0005, 0.0001, 0.00005, 0.00001])
momentum=instru.var.OrderedDiscrete([0.1, 0.3, 0.5, 0.7, 0.9])
patience=instru.var.OrderedDiscrete([2, 5, 8, 10])
latent_size=instru.var.OrderedDiscrete([64, 128, 256, 512])
n_hidden=instru.var.OrderedDiscrete([1, 2, 3, 4, 5])
hidden_size=instru.var.OrderedDiscrete([64, 128, 256, 512])
n_frames=instru.var.OrderedDiscrete([300, 400, 500, 600, 800])
dropout_prob=instru.var.OrderedDiscrete([0.1, 0.3, 0.5, 0.7])
model=instru.var.OrderedDiscrete(['resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'resnet_small', 'TDNN']) if args.model=='all' else args.model
ncoef=args.ncoef
epochs=args.epochs
batch_size=args.batch_size
valid_batch_size=args.valid_batch_size
n_workers=args.workers
cuda=args.cuda
train_hdf_file=args.train_hdf_file
data_info_path=args.data_info_path
valid_hdf_file=args.valid_hdf_file
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])

instrum=instru.Instrumentation(lr, l2, momentum, patience, latent_size, n_hidden, hidden_size, n_frames, model, ncoef, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, data_info_path, valid_hdf_file, checkpoint_path, softmax)

hp_optimizer=optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget, num_workers=args.hp_workers)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor))
