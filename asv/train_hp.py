from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_mining
import os
import sys
import pickle
from time import sleep

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(20)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

# Training settings
parser = argparse.ArgumentParser(description='Train for hp search')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--margin', type=float, default=0.3, metavar='m', help='margin fro triplet loss (default: 0.3)')
parser.add_argument('--lamb', type=float, default=0.001, metavar='l', help='Entropy regularization penalty (default: 0.001)')
parser.add_argument('--patience', type=int, default=10, metavar='S', help='Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)')
parser.add_argument('--model', choices=['resnet_lstm'], default='resnet_lstm', help='Model arch according to input type')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--latent-size', type=int, default=256, metavar='S', help='latent layer dimension (default: 256)')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--n-frames', type=int, default=800, metavar='N', help='maximum number of frames per utterance (default: 800)')
parser.add_argument('--n-cycles', type=int, default=3, metavar='N', help='cycles over speakers list to complete 1 epoch')
parser.add_argument('--valid-n-cycles', type=int, default=300, metavar='N', help='cycles over speakers list to complete 1 epoch')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--cuda', type=str, default=None)
parser.add_argument('--out-file', type=str, default='./eer.p')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--cp-name', type=str, default=None)
args = parser.parse_args()
args.cuda = True if args.cuda=='True' and torch.cuda.is_available() else False
args.swap = True if args.swap=='True' else False

if args.cuda:
	device = get_freer_gpu()
else:
	device = None

train_dataset = Loader_mining(hdf5_name = args.train_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=set_np_randomseed)

valid_dataset = Loader(hdf5_name = args.valid_hdf_file, max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=set_np_randomseed)

if args.model == 'resnet_lstm':
	model = model_.ResNet_lstm(n_z=args.latent_size, nh=args.n_hidden, n_h=args.hidden_size, proj_size=len(train_dataset.speakers_list), ncoef=args.ncoef, dropout_prob=args.dropout_prob, sm_type=args.softmax)

if args.cuda:
	model = model.cuda(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, margin=args.margin, lambda_=args.lamb, patience=args.patience, verbose=-1, device=device, cp_name=args.cp_name, save_cp=True, checkpoint_path=args.checkpoint_path, pretrain=False, cuda=args.cuda)

best_eer = trainer.train(n_epochs=args.epochs, save_every=args.epochs+10)

out_file = open(args.out_file, 'wb')
pickle.dump(best_eer[0], out_file)
out_file.close()
