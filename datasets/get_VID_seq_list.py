#!/usr/bin/python3
"""Script for creating text file containing sequences of 10 frames of particular video. Here we neglect all the frames where 
there is no object in it as it was done in the official implementation in tensorflow.
Global Variables
----------------
dirs : containing list of all the training dataset folders
dirs_val : containing path to val folder of dataset
dirs_test : containing path to test folder of dataset
"""
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

dirs = ['ILSVRC2015_VID_train_0000/',
		'ILSVRC2015_VID_train_0001/',
		'ILSVRC2015_VID_train_0002/',
		'ILSVRC2015_VID_train_0003/']
dirs_val = ['/media/sine/space/vikrant/ILSVRC2015/Data/VID/val/']
dirs_test = ['/media/sine/space/vikrant/ILSVRC2015/Data/VID/test/']
	



file_write_obj = open('train_VID_seqs_list.txt','w')
seq_list = []
with open('train_VID_list.txt') as f:
	for line in f:
		seq_list.append(line.rstrip())
for i in range(0,int(len(seq_list)/10)):
	seqs = ''
	for j in range(0,10):
		seqs = seqs + seq_list[10*i + j] + ','
	seqs = seqs[:-1]
	file_write_obj.writelines(seqs)
	file_write_obj.write('\n')
file_write_obj.close()
file_write_obj = open('val_VID_seqs_list.txt','w')
seq_list = []
with open('val_VID_list.txt') as f:
	for line in f:
		seq_list.append(line.rstrip())
for i in range(0,int(len(seq_list)/10)):
	#image_path = seq_list[10*i].split('/')[0]
	#seqs = image_path+'/'+':'
	seqs = ''
	for j in range(0,10):
		seqs = seqs + seq_list[10*i + j] + ','
	seqs = seqs[:-1] 
	file_write_obj.writelines(seqs)
	file_write_obj.write('\n')
file_write_obj.close()
file_write_obj = open('test_VID_seqs_list.txt','w')
for dir in dirs_test:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		image_list = np.sort(os.listdir(seq_path))
		for image in image_list:
			file_write_obj.writelines(seq+image)
			file_write_obj.write('\n')

file_write_obj.close()
