from concurrent import futures
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import subprocess
import shlex
import numpy as np
from time import sleep
import pickle
import glob
import torch
import os
import shutil

def get_file_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.p'

	while os.path.isfile(fname):
		fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.p'

	file_ = open(fname, 'wb')
	pickle.dump(None, file_)
	file_.close()

	return fname

def remove_err_out_files(id_):

	files_list = glob.glob('*'+id_+'*')

	for file_ in files_list:
		os.remove(file_)

# Training settings
parser=argparse.ArgumentParser(description='HP search for Imagenet')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--sub-file', type=str, default='./run_hp.sh', metavar='Path', help='Path to sge submission file')
parser.add_argument('--data-path', type=str, default='./data_train', metavar='Path', help='Path to data')
parser.add_argument('--hdf-path', type=str, default=None, metavar='Path', help='Path to data stored in hdf. Has priority over data path if set')
parser.add_argument('--valid-data-path', type=str, default='./data_val', metavar='Path', help='Path to data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to valid data stored in hdf. Has priority over valid data path if set')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--nclasses', type=int, default=1000, metavar='N', help='number of classes (default: 1000)')
parser.add_argument('--pretrained', action='store_true', default=False, help='Get pretrained weights on imagenet. Encoder only')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--temp-folder', type=str, default='temp', metavar='Path', help='Temp folder for pickle files')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()

args.cuda=True if not args.no_cuda else False

def train(lr, l2, momentum, smoothing, patience, model, emb_size, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, hdf_path, valid_data_path, valid_hdf_path, submission_file, checkpoint_path, softmax, n_classes, pretrained):

	file_name = get_file_name(tmp_dir)
	np.random.seed()

	command = 'sbatch' + ' ' + submission_file + ' ' + str(lr) + ' ' + str(l2) + ' ' + str(momentum) + ' ' + str(smoothing) + ' ' + str(int(patience)) + ' ' + str(model) + ' ' + str(int(emb_size)) + ' ' + str(int(n_hidden)) + ' ' + str(int(hidden_size)) + ' ' + str(dropout_prob) + ' ' + str(epochs) + ' ' + str(batch_size) + ' ' + str(valid_batch_size) + ' ' + str(n_workers) + ' ' + str(cuda) + ' ' + str(data_path) + ' ' + str(valid_data_path) + ' ' + str(hdf_path) + ' ' + str(valid_hdf_path) + ' ' + str(file_name) + ' ' + str(checkpoint_path) + ' ' + str(file_name.split('/')[-1]+'t')+ ' ' + str(softmax) + ' ' + str(n_classes) + ' ' + str(pretrained)

	for j in range(10):

		sleep(np.random.randint(10,120,1)[0])

		result=None

		p=subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

		out = p.communicate()

		job_id = out[0].decode('utf-8').split(' ')[3].strip()

		result_file = open(file_name, 'rb')
		result = pickle.load(result_file)
		result_file.close()

		if result is not None:
			remove_err_out_files(job_id)
			os.remove(file_name)

			print(' ')
			print('Best EER in result file ' + file_name.split('/')[-1].split('.p')[0] + ' was: {}'.format(result))
			print(' ')
			print('With hyperparameters:')
			print('Selected model: {}'.format(model))
			print('Embedding size: {}'.format(emb_size))
			print('Hidden layer size: {}'.format(hidden_size))
			print('Number of hidden layers: {}'.format(n_hidden))
			print('Dropout rate: {}'.format(dropout_prob))
			print('Batch size: {}'.format(batch_size))
			print('LR: {}'.format(lr))
			print('Momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('Label smoothing: {}'.format(smoothing))
			print('Patience: {}'.format(patience))
			print('Softmax Mode is: {}'.format(softmax))
			print('Pretrained: {}'.format(pretrained))
			print(' ')

			return result

	return 0.5

lr = instru.var.OrderedDiscrete([1e-2, 1e-3, 1e-4, 1e-5])
l2 = instru.var.OrderedDiscrete([1e-2, 1e-3, 1e-4, 1e-5])
momentum = instru.var.OrderedDiscrete([0.1, 0.5, 0.9])
smoothing=instru.var.OrderedDiscrete([0.0, 0.05, 0.1, 0.2])
patience = instru.var.OrderedDiscrete([3, 5, 10, 20])
model = args.model
emb_size = instru.var.OrderedDiscrete([128, 256, 350, 512])
n_hidden=instru.var.OrderedDiscrete([2, 3, 4, 5])
hidden_size=instru.var.OrderedDiscrete([128, 256, 350, 512])
dropout_prob=instru.var.OrderedDiscrete([0.01, 0.1, 0.2, 0.3])
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path if args.data_path else 'none'
hdf_path = args.hdf_path if args.hdf_path else 'none'
valid_data_path = args.valid_data_path if args.valid_data_path else 'none'
valid_hdf_path = args.valid_hdf_path if args.valid_hdf_path else 'none'
sub_file=args.sub_file
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])
n_classes = args.nclasses
pretrained = args.pretrained

tmp_dir = os.getcwd() + '/' + args.temp_folder + '/'

if not os.path.isdir(tmp_dir):
	os.mkdir(tmp_dir)

instrum = instru.Instrumentation(lr, l2, momentum, smoothing, patience, model, emb_size, n_hidden, hidden_size, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path, hdf_path, valid_hdf_path, sub_file, checkpoint_path, softmax, n_classes, pretrained)

hp_optimizer = optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor, verbosity=2))

shutil.rmtree(tmp_dir)