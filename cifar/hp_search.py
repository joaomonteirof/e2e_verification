import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
from data_load import Loader
import numpy as np
import os

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

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

def train(lr, l2, momentum, margin, lambda_, patience, swap, model, n_hidden, hidden_size, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, checkpoint_path):

	cp_name = get_cp_name(checkpoint_path)

	transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
	transform_test = transforms.ToTensor()

	#trainset = Loader(data_path)
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=n_workers, worker_init_fn=set_np_randomseed)

	#validset = Loader(valid_data_path)
	validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers)

	if model == 'vgg':
		model_ = vgg.VGG('VGG16', nh=int(n_hidden), n_h=int(hidden_size))
	elif model == 'resnet':
		model_ = resnet.ResNet18(nh=int(n_hidden), n_h=int(hidden_size))
	elif model == 'densenet':
		model_ = densenet.densenet_cifar(nh=int(n_hidden), n_h=int(hidden_size))

	if cuda:
		model_ = model_.cuda()

	optimizer = optim.SGD(model_.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

	trainer = TrainLoop(model_, optimizer, train_loader, valid_loader, margin=margin, lambda_=lambda_, patience=int(patience), verbose=-1, cp_name=cp_name, save_cp=True, checkpoint_path=checkpoint_path, swap=swap, cuda=cuda)

	for i in range(5):

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
			print('With hyperparameters:')
			print('Selected model: {}'.format(model))
			print('Hidden layer size size: {}'.format(int(hidden_size)))
			print('Number of hidden layers: {}'.format(int(n_hidden)))
			print('Batch size: {}'.format(batch_size))
			print('LR: {}'.format(lr))
			print('Momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('lambda: {}'.format(lambda_))
			print('Margin: {}'.format(margin))
			print('Swap: {}'.format(swap))
			print('Patience: {}'.format(patience))
			print(' ')

			return cost[0]
		except:
			pass

	print('Returning dummy cost due to failures while training.')
	return 0.99

lr = instru.var.Array(1).asfloat().bounded(1, 4).exponentiated(base=10, coeff=-1)
l2 = instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
momentum = instru.var.Array(1).asfloat().bounded(0.10, 0.95)
margin = instru.var.Array(1).asfloat().bounded(0.10, 1.00)
lambda_ = instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
patience = instru.var.Array(1).asfloat().bounded(1, 100)
swap = instru.var.OrderedDiscrete([True, False])
n_hidden=instru.var.Array(1).asfloat().bounded(1, 5)
hidden_size=instru.var.Array(1).asfloat().bounded(64, 512)
model = args.model
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path
valid_data_path = args.valid_data_path
checkpoint_path=args.checkpoint_path

instrum = instru.Instrumentation(lr, l2, momentum, margin, lambda_, patience, swap, model, n_hidden, hidden_size, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, checkpoint_path)

hp_optimizer = optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget)

print(hp_optimizer.optimize(train, verbosity=2))
