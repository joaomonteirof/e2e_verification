from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_valid
import os
import sys
import pickle
from time import sleep
from utils.utils import *
from utils.optimizer import TransformerOptimizer

# Training settings
parser = argparse.ArgumentParser(description='Train for hp search')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--b1', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--b2', type=float, default=0.98, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--model', choices=['resnet_stats', 'resnet_mfcc', 'resnet_lstm', 'resnet_small', 'resnet_large', 'TDNN'], default='resnet_lstm', help='Model arch according to input type')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--latent-size', type=int, default=256, metavar='S', help='latent layer dimension (default: 256)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--n-frames', type=int, default=800, metavar='N', help='maximum number of frames per utterance (default: 800)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--cuda', type=str, default=None)
parser.add_argument('--out-file', type=str, default='./eer.p')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--cp-name', type=str, default=None)
args = parser.parse_args()
args.cuda = True if args.cuda=='True' and torch.cuda.is_available() else False

train_dataset = Loader(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

valid_dataset = Loader_valid(hdf5_name = args.valid_hdf_file, max_nb_frames = args.n_frames)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

if args.model == 'resnet_stats':
	model = model_.ResNet_stats(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)
elif args.model == 'resnet_mfcc':
	model = model_.ResNet_mfcc(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)
if args.model == 'resnet_lstm':
	model = model_.ResNet_lstm(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)
elif args.model == 'resnet_small':
	model = model_.ResNet_small(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)
elif args.model == 'resnet_large':
	model = model_.ResNet_large(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)
elif args.model == 'TDNN':
	model = model_.TDNN(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=train_dataset.n_speakers, ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)

if args.cuda:
	device = get_freer_gpu()
else:
	device = None

if args.cuda:
	model = model.cuda(device)

optimizer = TransformerOptimizer(optim.Adam(model.parameters(), betas=(args.b1, args.b2), weight_decay=args.l2), lr=args.lr, warmup_steps=args.warmup)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, label_smoothing=args.smoothing, verbose=-1, device=device, cp_name=args.cp_name, save_cp=True, checkpoint_path=args.checkpoint_path, pretrain=False, cuda=args.cuda)

print(' ')
print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Pretrain Mode: {}'.format(args.pretrain))
print('Selected model: {}'.format(args.model))
print('Softmax Mode is: {}'.format(args.softmax))
print('Embeddings size: {}'.format(args.latent_size))
print('Number of hidden layers: {}'.format(args.n_hidden))
print('Size of hidden layers: {}'.format(args.hidden_size))
print('Batch size: {}'.format(args.batch_size))
print('Valid batch size: {}'.format(args.valid_batch_size))
print('LR: {}'.format(args.lr))
print('B1 and B2: {}, {}'.format(args.b1, args.b2))
print('l2: {}'.format(args.l2))
print('Warmup iterations: {}'.format(args.warmup))
print('Label smoothing: {}'.format(args.smoothing))
print('Max length: {}'.format(args.n_frames))
print('Number of train speakers: {}'.format(train_dataset.n_speakers))
print('Number of train examples: {}'.format(len(train_dataset.utt_list)))
if args.valid_hdf_file:
	print('Number of valid speakers: {}'.format(valid_dataset.n_speakers))
	print('Number of valid examples: {}'.format(len(valid_dataset.utt_list)))
print(' ')

best_eer = trainer.train(n_epochs=args.epochs, save_every=args.epochs+10)

out_file = open(args.out_file, 'wb')
pickle.dump(best_eer[0], out_file)
out_file.close()
