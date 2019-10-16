#!/usr/bin/python3
"""Script for creating text file containing sequences of all the video frames. Here we neglect all the frames where 
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
	



file_write_obj = open('train_VID_list.txt','w')
for dir in dirs:
	seqs = np.sort(os.listdir(os.path.join('/media/sine/space/vikrant/ILSVRC2015/Data/VID/train/'+dir)))
	for seq in seqs:
		seq_path = os.path.join('/media/sine/space/vikrant/ILSVRC2015/Data/VID/train/',dir,seq)
		relative_path = dir + seq
		image_list = np.sort(os.listdir(seq_path))
		count = 0
		for image in image_list:
			image_id = image.split('.')[0]
			anno_file = image_id + '.xml'
			anno_path = os.path.join('/media/sine/space/vikrant/ILSVRC2015/Annotations/VID/train/',dir,seq,anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0: # discarding images without object
				continue
			else:
				count = count + 1
				file_write_obj.writelines(relative_path+'/'+image_id)
				file_write_obj.write('\n')
file_write_obj.close()
file_write_obj = open('val_VID_list.txt','w')
for dir in dirs_val:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		image_list = np.sort(os.listdir(seq_path))
		count = 0
		for image in image_list:
			image_id = image.split('.')[0]
			anno_file = image_id + '.xml'
			anno_path = os.path.join('/media/sine/space/vikrant/ILSVRC2015/Annotations/VID/val/',seq,anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0:
				continue
			else:
				count = count + 1
				if count<=20:
					file_write_obj.writelines(seq+'/'+image_id)
					file_write_obj.write('\n')


file_write_obj.close()
# file_write_obj = open('test_VID_list.txt','w')
# for dir in dirs_test:
# 	seqs = np.sort(os.listdir(dir))
# 	for seq in seqs:
# 		seq_path = os.path.join(dir,seq)
# 		image_list = np.sort(os.listdir(seq_path))
# 		for image in image_list:
# 			file_write_obj.writelines(seq+image)
# 			file_write_obj.write('\n')

# file_write_obj.close()
