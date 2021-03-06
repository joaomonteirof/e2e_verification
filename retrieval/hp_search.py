import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from torch.utils.data import DataLoader
from data_load import Loader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os
import sys
from time import sleep
from torch.utils.tensorboard import SummaryWriter
from optimizer import TransformerOptimizer
from utils import *

def get_cp_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	return fname.split('/')[-1]

# Training settings
parser = argparse.ArgumentParser(description='Image retrieval')
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--data-path', type=str, default=None, metavar='Path', help='Path to data')
parser.add_argument('--hdf-path', type=str, default=None, metavar='Path', help='Path to data stored in hdf. Has priority over data path if set')
parser.add_argument('--valid-data-path', type=str, default=None, metavar='Path', help='Path to data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to valid data stored in hdf. Has priority over valid data path if set')
parser.add_argument('--stats', choices=['cars', 'cub', 'sop', 'imagenet'], default='imagenet')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--budget', type=int, default=100, metavar='N', help='Maximum training runs')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
parser.add_argument('--pretrained-path', type=str, nargs='+', default=[], metavar='Path', help='Paths to trained model. Discards output layers')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for logs')
parser.add_argument('--eval-every', type=int, default=1000, metavar='N', help='how many iterations to wait before evaluatiing models. Default is 1000')
parser.add_argument('--ablation', action='store_true', default=False, help='Drops the multi class classification loss')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

print(args,'\n')

def train(lr, l2, momentum, smoothing, warmup, model, emb_size, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, hdf_path, valid_hdf_path, checkpoint_path, softmax, pretrained, pretrained_path, max_gnorm, stats, log_dir, eval_every, ablation):

	args_dict = locals()

	cp_name = get_cp_name(checkpoint_path)

	if pretrained_path != 'none':
		print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
		ckpt=torch.load(pretrained_path, map_location = lambda storage, loc: storage)
		dropout_prob, n_hidden, hidden_size, emb_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['emb_size']
		if 'r_proj_size' in ckpt:
			rproj_size = ckpt['r_proj_size']
		else:
			rproj_size = -1
		print('\nUsing pretrained config for discriminator. Ignoring args.')

	args_dict['dropout_prob'], args_dict['n_hidden'], args_dict['hidden_size'], args_dict['emb_size'] = dropout_prob, n_hidden, hidden_size, emb_size

	if log_dir != 'none':
		writer = SummaryWriter(log_dir=os.path.join(log_dir, cp_name), comment=model, purge_step=0)
		writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':0.0})
	else:
		writer = None

	if stats=='cars':
		mean, std = [0.4461, 0.4329, 0.4345], [0.2888, 0.2873, 0.2946]
	elif stats=='cub':
		mean, std = [0.4782, 0.4925, 0.4418], [0.2330, 0.2296, 0.2647]
	elif stats=='sop':
		mean, std = [0.5603, 0.5155, 0.4796], [0.2939, 0.2991, 0.3085]
	elif stats=='imagenet':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

	if hdf_path != 'none':
		transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomPerspective(p=0.1), transforms.RandomGrayscale(p=0.1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
		trainset = Loader(hdf_path, transform_train)
	else:
		transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomPerspective(p=0.1), transforms.RandomGrayscale(p=0.1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])	
		trainset = datasets.ImageFolder(data_path, transform=transform_train)

	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

	if valid_hdf_path != 'none':
		transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
		validset = Loader(args.valid_hdf_path, transform_test)
	else:
		transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
		validset = datasets.ImageFolder(args.valid_data_path, transform=transform_test)
		
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

	nclasses = trainset.n_classes if isinstance(trainset, Loader) else len(trainset.classes)

	if model == 'vgg':
		model_ = vgg.VGG('VGG19', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=nclasses, emb_size=emb_size, r_proj_size=rproj_size)
	elif model == 'resnet':
		model_ = resnet.ResNet50(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=nclasses, emb_size=emb_size, r_proj_size=rproj_size)
	elif model == 'densenet':
		model_ = densenet.DenseNet121(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax, n_classes=nclasses, emb_size=emb_size, r_proj_size=rproj_size)

	if pretrained_path != 'none':
		if ckpt['sm_type'] == 'am_softmax':
			del(ckpt['model_state']['out_proj.w'])
		elif ckpt['sm_type'] == 'softmax':
			del(ckpt['model_state']['out_proj.w.weight'])
			del(ckpt['model_state']['out_proj.w.bias'])

		print(model_.load_state_dict(ckpt['model_state'], strict=False))
		print('\n')

	if pretrained:
		print('\nLoading pretrained encoder from torchvision\n')
		if model == 'vgg':
			model_pretrained = torchvision.models.vgg19(pretrained=True)
		elif model == 'resnet':
			model_pretrained = torchvision.models.resnet50(pretrained=True)
		elif model == 'densenet':
			model_pretrained = torchvision.models.densenet121(pretrained=True)

		print(model_.load_state_dict(model_pretrained.state_dict(), strict=False))
		print('\n')

	if cuda:
		device = get_freer_gpu()
		model_ = model_.cuda(device)
		torch.backends.cudnn.benchmark=True

	optimizer = TransformerOptimizer(optim.SGD(model_.parameters(), lr=lr, momentum=momentum, weight_decay=l2, nesterov=True), lr=lr, warmup_steps=warmup)

	trainer = TrainLoop(model_, optimizer, train_loader, valid_loader, max_gnorm=max_gnorm, label_smoothing=smoothing, verbose=-1, cp_name=cp_name, save_cp=True, checkpoint_path=checkpoint_path, ablation=ablation, cuda=cuda, logger=writer)

	for i in range(5):

		print(' ')
		print('Hyperparameters:')
		print('Selected model: {}'.format(model))
		print('Embedding size: {}'.format(emb_size))
		print('Hidden layer size: {}'.format(hidden_size))
		print('Number of hidden layers: {}'.format(n_hidden))
		print('Random projection size: {}'.format(rproj_size))
		print('Dropout rate: {}'.format(dropout_prob))
		print('Batch size: {}'.format(batch_size))
		print('LR: {}'.format(lr))
		print('Momentum: {}'.format(momentum))
		print('l2: {}'.format(l2))
		print('Label smoothing: {}'.format(smoothing))
		print('Warmup iterations: {}'.format(warmup))
		print('Softmax Mode is: {}'.format(softmax))
		print('Pretrained: {}'.format(pretrained))
		print('Pretrained path: {}'.format(pretrained_path))
		print('Evaluate every {} iterations.'.format(eval_every))
		print('Ablation Mode: {}'.format(ablation))
		print(' ')

		if i>0:
			print(' ')
			print('Trial {}'.format(i+1))
			print(' ')

		try:
			cost = trainer.train(n_epochs=epochs, save_every=epochs+10, eval_every=eval_every)

			print(' ')
			print('Best e2e EER in file ' + cp_name + ' was: {}'.format(cost[0]))
			print('Best cos EER in file ' + cp_name + ' was: {}'.format(cost[1]))
			print(' ')

			if log_dir != 'none':
				writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':cost[0]})

			return cost[0]
		except:
			print("Error:", sys.exc_info())
			pass

	print('Returning dummy cost due to failures while training.')
	return 0.99

lr = instru.var.Array(1).asfloat().bounded(1e-3, 1.00)
l2 = instru.var.Array(1).asfloat().bounded(1e-5, 1e-3)
momentum = instru.var.Array(1).asfloat().bounded(0.01, 0.999)
smoothing=instru.var.OrderedDiscrete([0.0, 0.05, 0.1, 0.2])
warmup = instru.var.OrderedDiscrete([1, 1000, 5000, 10000])
model = args.model
emb_size = instru.var.OrderedDiscrete([128, 256, 350, 512]) if len(args.pretrained_path) == 0 else 1
n_hidden=instru.var.OrderedDiscrete([2, 3, 4, 5]) if len(args.pretrained_path) == 0 else 1
hidden_size=instru.var.OrderedDiscrete([128, 256, 350, 512]) if len(args.pretrained_path) == 0 else 1
dropout_prob=instru.var.OrderedDiscrete([0.01, 0.1, 0.2, 0.3]) if len(args.pretrained_path) == 0 else 1
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path if args.data_path is not None else 'none'
hdf_path = args.hdf_path if args.hdf_path is not None else 'none'
valid_data_path = args.valid_data_path if args.valid_data_path is not None else 'none'
valid_hdf_path = args.valid_hdf_path if args.valid_hdf_path is not None else 'none'
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])
pretrained = args.pretrained
if len(args.pretrained_path) > 1:
	pretrained_path = instru.var.OrderedDiscrete(args.pretrained_path)
elif len(args.pretrained_path) == 1:
	pretrained_path = args.pretrained_path
else:
	pretrained_path = 'none'
max_gnorm = instru.var.OrderedDiscrete([10, 50, 100])
stats = args.stats
log_dir = args.logdir if args.logdir else 'none'
eval_every = args.eval_every
ablation = args.ablation

instrum = instru.Instrumentation(lr, l2, momentum, smoothing, warmup, model, emb_size, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, hdf_path, valid_hdf_path, checkpoint_path, softmax, pretrained, pretrained_path, max_gnorm, stats, log_dir, eval_every, ablation)

hp_optimizer = optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget)

print(hp_optimizer.optimize(train, verbosity=2))
