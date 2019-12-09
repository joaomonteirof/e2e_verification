import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from time import sleep

from utils import *

def get_cp_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	return fname.split('/')[-1]

# Training settings
parser = argparse.ArgumentParser(description='Cifar10 Classification')
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--data-path', type=str, default='./data/cifar10_train_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/cifar10_test_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--budget', type=int, default=100, metavar='N', help='Maximum training runs')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--checkpoint-path', type=str, default='./', metavar='Path', help='Path for checkpointing')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, smoothing, patience, model, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, checkpoint_path, softmax):

	cp_name = get_cp_name(checkpoint_path)

	transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])

	#trainset = Loader(data_path)
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers, worker_init_fn=set_np_randomseed)

	#validset = Loader(valid_data_path)
	validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True, num_workers=n_workers)

	if model == 'vgg':
		model_ = vgg.VGG('VGG16', nh=int(n_hidden), n_h=int(hidden_size), dropout_prob=dropout_prob, sm_type=softmax)
	elif model == 'resnet':
		model_ = resnet.ResNet18(nh=int(n_hidden), n_h=int(hidden_size), dropout_prob=dropout_prob, sm_type=softmax)
	elif model == 'densenet':
		model_ = densenet.densenet_cifar(nh=int(n_hidden), n_h=int(hidden_size), dropout_prob=dropout_prob, sm_type=softmax)

	if args.cuda:
		device = get_freer_gpu()
		model_ = model_.cuda(device)

	optimizer = optim.SGD(model_.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

	trainer = TrainLoop(model_, optimizer, train_loader, valid_loader, patience=int(patience), label_smoothing=smoothing, verbose=-1, cp_name=cp_name, save_cp=True, checkpoint_path=checkpoint_path, cuda=cuda)

	for i in range(5):

		print(' ')
		print('Hyperparameters:')
		print('Selected model: {}'.format(model))
		print('Hidden layer size size: {}'.format(int(hidden_size)))
		print('Number of hidden layers: {}'.format(int(n_hidden)))
		print('Dropout rate: {}'.format(dropout_prob))
		print('Batch size: {}'.format(batch_size))
		print('LR: {}'.format(lr))
		print('Momentum: {}'.format(momentum))
		print('l2: {}'.format(l2))
		print('Label smoothing: {}'.format(smoothing))
		print('Patience: {}'.format(patience))
		print('Softmax Mode is: {}'.format(softmax))
		print(' ')

		if i>0:
			print(' ')
			print('Trial {}'.format(i+1))
			print(' ')

		try:
			cost = trainer.train(n_epochs=epochs, save_every=epochs+10)
			print(' ')
			print('Best e2e EER in file ' + cp_name + ' was: {}'.format(cost[0]))
			print('Best cos EER in file ' + cp_name + ' was: {}'.format(cost[1]))
			print('Best Error Rate in file ' + cp_name + ' was: {}'.format(cost[2]))
			print(' ')

			return cost[0]

		except:
			print("Error:", sys.exc_info())
			pass

	print('Returning dummy cost due to failures while training.')
	return 0.99

lr = instru.var.OrderedDiscrete([0.5, 0.1, 0.01, 0.001])
l2 = instru.var.OrderedDiscrete([1e-2, 1e-3, 1e-4, 1e-5])
momentum = instru.var.OrderedDiscrete([0.1, 0.5, 0.9])
smoothing=instru.var.OrderedDiscrete([0.01, 0.1, 0.2])
patience = instru.var.OrderedDiscrete([1, 5, 10, 20])
n_hidden=instru.var.OrderedDiscrete([2, 3, 4, 5])
hidden_size=instru.var.OrderedDiscrete([128, 256, 350, 512])
dropout_prob=instru.var.OrderedDiscrete([0.01, 0.1, 0.2])
model = args.model
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path
valid_data_path = args.valid_data_path
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])

instrum = instru.Instrumentation(lr, l2, momentum, smoothing, patience, model, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, checkpoint_path, softmax)

hp_optimizer = optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget)

print(hp_optimizer.optimize(train, verbosity=2))
