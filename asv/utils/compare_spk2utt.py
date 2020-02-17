import argparse
import os

def read_file(path):
	with open(path, 'r') as file:
		lines = file.readlines()

	spk_list = []

	for line in lines:
		spk = line.split(' ')[0]
		spk_list.append(spk)

	return spk_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Check whether speaker lists in distinct spk2utt files are disjoint')
	parser.add_argument('--file1', type=str, required=True, help='Path to first file')
	parser.add_argument('--file2', type=str, required=True, help='Path to first file')
	args = parser.parse_args()

	list1 = read_file(args.file1)
	list2 = read_file(args.file2)

	overlap_count = 0

	for el in list1:
		if el in list2:
			overlap_count+=1

	print('\nNumber of lines in file 1: {}'.format(len(list1)))
	print('\nNumber of lines in file 2: {}'.format(len(list2)))
	print('\nNumber of overlaps: {}'.format(overlap_count))