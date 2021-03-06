from __future__ import print_function
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
from data_load import Loader
import numpy as np
from time import sleep
import os
import sys
from optimizer import TransformerOptimizer
from torch.utils.tensorboard import SummaryWriter

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(2)
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
parser = argparse.ArgumentParser(description='Image retrieval')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--l2', type=float, default=1e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data_train', metavar='Path', help='Path to data')
parser.add_argument('--hdf-path', type=str, default=None, metavar='Path', help='Path to data stored in hdf. Has priority over data path if set')
parser.add_argument('--valid-data-path', type=str, default='./data_val', metavar='Path', help='Path to data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to valid data stored in hdf. Has priority over valid data path if set')
parser.add_argument('--stats', choices=['cars', 'cub', 'sop', 'imagenet'], default='imagenet')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--softmax', choices=['softmax', 'am_softmax'], default='softmax', help='Softmax type')
parser.add_argument('--emb-size', type=int, default=256, metavar='N', help='Embedding dimension (default: 256)')
parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path to trained model. Discards outpu layer')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
parser.add_argument('--rproj-size', type=int, default=-1, metavar='S', help='Random projection size - active if greater than 1')
parser.add_argument('--dropout-prob', type=float, default=0.25, metavar='p', help='Dropout probability (default: 0.25)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before saving checkpoints. Default is 1')
parser.add_argument('--eval-every', type=int, default=1000, metavar='N', help='how many iterations to wait before evaluatiing models. Default is 1000')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
parser.add_argument('--ablation', action='store_true', default=False, help='Drops the multi class classification loss')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.stats=='cars':
	mean, std = [0.4461, 0.4329, 0.4345], [0.2888, 0.2873, 0.2946]
elif args.stats=='cub':
	mean, std = [0.4782, 0.4925, 0.4418], [0.2330, 0.2296, 0.2647]
elif args.stats=='sop':
	mean, std = [0.5603, 0.5155, 0.4796], [0.2939, 0.2991, 0.3085]
elif args.stats=='imagenet':
	mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

if args.cuda:
	torch.backends.cudnn.benchmark=True

if args.hdf_path:
	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomPerspective(p=0.1), transforms.RandomGrayscale(p=0.1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	trainset = Loader(args.hdf_path, transform_train)
else:
	transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomPerspective(p=0.1), transforms.RandomGrayscale(p=0.1), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])	
	trainset = datasets.ImageFolder(args.data_path, transform=transform_train)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

if args.valid_hdf_path:
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = Loader(args.valid_hdf_path, transform_test)
else:
	transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
	validset = datasets.ImageFolder(args.valid_data_path, transform=transform_test)
	
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

args.nclasses = trainset.n_classes if isinstance(trainset, Loader) else len(trainset.classes)

print(args, '\n')

if args.pretrained_path:
	print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
	ckpt=torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
	args.dropout_prob, args.n_hidden, args.hidden_size, args.emb_size = ckpt['dropout_prob'], ckpt['n_hidden'], ckpt['hidden_size'], ckpt['emb_size']
	if 'r_proj_size' in ckpt:
		args.rproj_size = ckpt['r_proj_size']
	print('\nUsing pretrained config for discriminator. Ignoring args.')

if args.model == 'vgg':
	model = vgg.VGG('VGG19', nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, n_classes=args.nclasses, emb_size=args.emb_size, r_proj_size=args.rproj_size)
elif args.model == 'resnet':
	model = resnet.ResNet50(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, n_classes=args.nclasses, emb_size=args.emb_size, r_proj_size=args.rproj_size)
elif args.model == 'densenet':
	model = densenet.DenseNet121(nh=args.n_hidden, n_h=args.hidden_size, dropout_prob=args.dropout_prob, sm_type=args.softmax, n_classes=args.nclasses, emb_size=args.emb_size, r_proj_size=args.rproj_size)

if args.pretrained_path:
	if ckpt['sm_type'] == 'am_softmax':
		del(ckpt['model_state']['out_proj.w'])
	elif ckpt['sm_type'] == 'softmax':
		del(ckpt['model_state']['out_proj.w.weight'])
		del(ckpt['model_state']['out_proj.w.bias'])

	print(model.load_state_dict(ckpt['model_state'], strict=False))
	print('\n')

elif args.pretrained:
	print('\nLoading pretrained encoder from torchvision\n')
	if args.model == 'vgg':
		model_pretrained = torchvision.models.vgg19(pretrained=True)
	elif args.model == 'resnet':
		model_pretrained = torchvision.models.resnet50(pretrained=True)
	elif args.model == 'densenet':
		model_pretrained = torchvision.models.densenet121(pretrained=True)

	print(model.load_state_dict(model_pretrained.state_dict(), strict=False))
	print('\n')

if args.verbose >0:
	print(model)

if args.cuda:
	device = get_freer_gpu()
	model = model.to(device)

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=0 if args.checkpoint_epoch is None else int(args.checkpoint_epoch*len(train_loader)))
	args_dict = parse_args_for_log(args)
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_eer':0.0})
else:
	writer = None

optimizer = TransformerOptimizer(optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True), lr=args.lr, warmup_steps=args.warmup)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, label_smoothing=args.smoothing, verbose=args.verbose, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, ablation=args.ablation, cuda=args.cuda, logger=writer)

if args.verbose >0:
	print('\nCuda Mode is: {}'.format(args.cuda))
	print('Selected model: {}'.format(args.model))
	print('Batch size: {}'.format(args.batch_size))
	print('LR: {}'.format(args.lr))
	print('Momentum: {}'.format(args.momentum))
	print('l2: {}'.format(args.l2))
	print('Label smoothing: {}'.format(args.smoothing))
	print('Warmup iterations: {}'.format(args.warmup))
	print('Max. grad norm: {}'.format(args.max_gnorm))
	print('Dropout rate: {}'.format(args.dropout_prob))
	print('Softmax Mode is: {}'.format(args.softmax))
	print('Embedding dimension: {}'.format(args.emb_size))
	print('Number of hidden layers: {}'.format(args.n_hidden))
	print('Size of hidden layers: {}'.format(args.hidden_size))
	print('Random projection size: {}'.format(args.rproj_size))
	print('Stats: {}'.format(args.stats))
	print('Ablation Mode: {}'.format(args.ablation))

trainer.train(n_epochs=args.epochs, save_every=args.save_every, eval_every=args.eval_every)
