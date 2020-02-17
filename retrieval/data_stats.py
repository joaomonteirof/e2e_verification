import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Get data stats')
	parser.add_argument('--path-to-data', type=str, default='./data/', metavar='Path', help='Path to scp files with features')
	parser.add_argument('--batch-size', type=int, default=1000, metavar='N', help='Batch size')
	parser.add_argument('--sample-size', type=int, default=10000, metavar='N', help='Sample size to estimate data stats')
	parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
	args = parser.parse_args()

	transform = transforms.Compose([transforms.ToTensor()])
	dataset = datasets.ImageFolder(args.path_to_data, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

	data = []
	data_iterator = tqdm(enumerate(dataloader), total=len(dataloader))

	for i, batch in data_iterator:

		x, _ = batch

		data.append(x)

		if (i+1)*args.batch_size>args.sample_size:
			break

	data = torch.cat(data, 0)

	mean, var, std = data.mean((0,2,3)), data.var((0,2,3)), data.std((0,2,3))

	print('\n'+args.path_to_data+'\n')
	print('mean: {}'.format(mean))
	print('var: {}'.format(var))
	print('std: {}'.format(std))
	print('\n')