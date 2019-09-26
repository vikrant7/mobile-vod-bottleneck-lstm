import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

def _read_image_seq_ids(image_sets_file):
		seq_list = []
		with open(image_sets_file) as f:
			for line in f:
				seq_list.append(line.rstrip())
		return seq_list

if __name__ == "__main__":
	image_sets_file = "train_VID_seqs_list.txt"
	seq_list = _read_image_seq_ids(image_sets_file)
	image_seq = seq_list[0]
	print(image_seq)
	image_path = image_seq.split(';')[0]
	image_ids = image_seq.split(';')[1].split(',')

	print(image_path)
	print(image_ids)
	print(len(image_ids))