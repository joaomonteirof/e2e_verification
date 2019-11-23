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

def kill_job(id_):

	try:
		status = subprocess.check_output('qdel ' + id_, shell=True)
		print(' ')
		print('Job {} killed'.format(id_))
		print(' ')
	except:
		pass

def remove_err_out_files(id_):
	files_list = glob.glob('*.'+id_+'.*')
	for file_ in files_list:
		os.remove(file_)

def check_job_running(id_):

	try:
		status = subprocess.check_output('qstat -s rp | grep ' + id_, shell=True)
		return True
	except:
		return False

# Training settings
parser=argparse.ArgumentParser(description='HP search for ASV')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--budget', type=int, default=30, metavar='N', help='Maximum training runs')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--sub-file', type=str, default='./run_hp.sh', metavar='Path', help='Path to sge submission file')
parser.add_argument('--train-hdf-file', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-file', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--model', choices=['resnet_stats', 'resnet_mfcc', 'resnet_lstm', 'resnet_small', 'resnet_large', 'TDNN', 'all'], default='resnet_lstm', help='Model arch according to input type')
parser.add_argument('--ndiscriminators', type=int, default=1, metavar='N', help='number of discriminators (default: 1)')
parser.add_argument('--rproj', action='store_true', default=False, help='Enable search for random projection size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--hp-workers', type=int, help='number of search workers', default=1)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--ncoef', type=int, default=23, metavar='N', help='number of MFCCs (default: 23)')
parser.add_argument('--temp-folder', type=str, default='temp', metavar='Path', help='Temp folder for pickle files')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args=parser.parse_args()
args.cuda=True if not args.no_cuda else False

def train(lr, l2, momentum, smoothing, warmup, latent_size, n_hidden, hidden_size, n_frames, model, ndiscriminators, rproj_size, ncoef, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, submission_file, tmp_dir, cp_path, softmax, max_gnorm, logdir):

	file_name = get_file_name(tmp_dir)
	np.random.seed()

	if rproj_size>0:
		rproj_size = int(rproj_size*latent_size*2)
	else:
		rproj_size = -1

	command = 'qsub' + ' ' + submission_file + ' ' + str(lr) + ' ' + str(l2) + ' ' + str(momentum) + ' ' + str(smoothing) + ' ' + str(int(warmup)) + ' ' + str(int(latent_size)) + ' ' + str(int(n_hidden)) + ' ' + str(int(hidden_size)) + ' ' + str(int(n_frames)) + ' ' + str(model) + ' ' + str(ndiscriminators) + ' ' + str(rproj_size) + ' ' + str(ncoef) + ' ' + str(dropout_prob) + ' ' + str(epochs) + ' ' + str(batch_size) + ' ' + str(valid_batch_size) + ' ' + str(n_workers) + ' ' + str(cuda) + ' ' + str(train_hdf_file) + ' ' + str(valid_hdf_file) + ' ' + str(file_name) + ' ' + str(cp_path) + ' ' + str(file_name.split('/')[-1]+'t') + ' ' + str(softmax) + ' ' + str(max_gnorm) + ' ' + str(logdir)

	for j in range(10):

		sleep(np.random.randint(10,120,1)[0])

		result=None

		p=subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

		out = p.communicate()
		job_id = out[0].decode('utf-8').split(' ')[2]

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
			print('Model: {}'.format(model))
			print('Number of discriminators: {}'.format(ndiscriminators))
			print('Random projection size: {}'.format(rproj_size))
			print('Softmax mode: {}'.format(softmax))
			print('Embeddings size: {}'.format(int(latent_size)))
			print('Number of hidden layers: {}'.format(int(n_hidden)))
			print('Size of hidden layers: {}'.format(int(hidden_size)))
			print('Dropout rate: {}'.format(dropout_prob))
			print('LR: {}'.format(lr))
			print('Momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('Max. grad norm: {}'.format(max_gnorm))
			print('Warmup iterations: {}'.format(warmup))
			print('Label smoothing: {}'.format(smoothing))
			print('Max. number of frames: {}'.format(int(n_frames)))
			print(' ')

			return result

	return 0.5

lr=instru.var.OrderedDiscrete([1.0, 0.5, 0.1, 0.01])
l2=instru.var.OrderedDiscrete([0.001, 0.0001, 0.00001])
momentum=instru.var.OrderedDiscrete([0.7, 0.85, 0.95])
smoothing=instru.var.OrderedDiscrete([0.0, 0.1, 0.2])
warmup=instru.var.OrderedDiscrete([1, 500, 2000])
latent_size=instru.var.OrderedDiscrete([128, 256, 512])
n_hidden=instru.var.OrderedDiscrete([1, 2, 3, 4])
hidden_size=instru.var.OrderedDiscrete([64, 128, 256, 512])
n_frames=instru.var.OrderedDiscrete([300, 500, 800])
dropout_prob=instru.var.OrderedDiscrete([0.01, 0.1, 0.2])
model=instru.var.OrderedDiscrete(['resnet_mfcc', 'resnet_lstm', 'resnet_stats', 'resnet_small', 'TDNN']) if args.model=='all' else args.model
ndiscriminators=args.ndiscriminators
rproj_size=instru.var.OrderedDiscrete([-1.0, 0.3, 0.5, 0.8]) if args.rproj else -1
ncoef=args.ncoef
epochs=args.epochs
batch_size=args.batch_size
valid_batch_size=args.valid_batch_size
n_workers=args.workers
cuda=args.cuda
train_hdf_file=args.train_hdf_file
valid_hdf_file=args.valid_hdf_file
sub_file=args.sub_file
checkpoint_path=args.checkpoint_path
softmax=instru.var.OrderedDiscrete(['softmax', 'am_softmax'])
max_gnorm=instru.var.OrderedDiscrete([10.0, 20.0, 50.0])
logdir=args.logdir

tmp_dir = os.getcwd() + '/' + args.temp_folder + '/'

if not os.path.isdir(tmp_dir):
	os.mkdir(tmp_dir)

instrum=instru.Instrumentation(lr, l2, b1, b2, smoothing, warmup, latent_size, n_hidden, hidden_size, n_frames, model, ndiscriminators, rproj_size, ncoef, dropout_prob, epochs, batch_size, valid_batch_size, n_workers, cuda, train_hdf_file, valid_hdf_file, sub_file, tmp_dir, checkpoint_path, softmax, max_gnorm, logdir)

hp_optimizer=optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget, num_workers=args.hp_workers)

with futures.ThreadPoolExecutor(max_workers=args.hp_workers) as executor:
	print(hp_optimizer.optimize(train, executor=executor, verbosity=2))

shutil.rmtree(tmp_dir)
