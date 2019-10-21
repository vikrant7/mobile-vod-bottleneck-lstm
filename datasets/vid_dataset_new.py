#!/usr/bin/python3
"""Script for creating dataset of VID data. Here we have two classes: one for sequencial dataset preparation
and other for normal object localization and classification task.
Classes
----------------
VIDDataset : class for loading dataset in sequences of 10 consecutive video frames
ImagenetDataset : class for loading dataset single frame at a time
"""
import pickle
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VIDDataset:

	def __init__(self, root, transform=None, target_transform=None, is_val=False,is_test=False, label_file=None):
		"""Dataset for VID data.
		Args:
			root: the root of the ILSVRC2015 dataset, the directory contains the following sub-directories:
				Annotations, ImageSets, Data
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		self.target_transform = target_transform
		self.is_test = is_test
		self.is_val = is_val
		if is_test:
			image_sets_file = "datasets/test_VID_seqs_list.txt"
		elif is_val:
			image_sets_file = "datasets/val_VID_seqs_list.txt"
		else:
			image_sets_file = "datasets/train_VID_seqs_list.txt"
		self.seq_list = VIDDataset._read_image_seq_ids(image_sets_file)

		logging.info("using default Imagenet VID classes.")
		self._classes_names = ['__background__',  # always index 0
					'airplane', 'antelope', 'bear', 'bicycle',
					'bird', 'bus', 'car', 'cattle',
					'dog', 'domestic_cat', 'elephant', 'fox',
					'giant_panda', 'hamster', 'horse', 'lion',
					'lizard', 'monkey', 'motorcycle', 'rabbit',
					'red_panda', 'sheep', 'snake', 'squirrel',
					'tiger', 'train', 'turtle', 'watercraft',
					'whale', 'zebra']
		self._classes_map = ['__background__',  # always index 0
						'n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049']

		self._name_to_class = {self._classes_map[i]: self._classes_names[i] for i in range(len(self._classes_names))}
		self._class_to_ind = {class_name: i for i, class_name in enumerate(self._classes_names)}

	def __getitem__(self, index):
		if self.is_test:
			image_seq = self.seq_list[index]
			image_file = self.root / f"Data/VID/test/{image_seq}"
			image = cv2.imread(str(image_file))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			return image
		else:
			images = []
			boxes_seq = []
			labels_seq = []
			image_seq = self.seq_list[index]
			#image_path = image_seq.split(':')[0]
			image_ids = image_seq.split(',')
			for image_id in image_ids:
				if self.is_val:
					annotation_file = os.path.join(self.root,f"Annotations/VID/val/{image_id}.xml")
					image_file = os.path.join(self.root , f"Data/VID/val/{image_id}.JPEG")
				else:
					annotation_file = self.root / f"Annotations/VID/train/{image_id}.xml"
					image_file = self.root / f"Data/VID/train/{image_id}.JPEG"
				image = cv2.imread(str(image_file))
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				objects = ET.parse(annotation_file).findall("object")
				boxes = []
				labels = []
				if len(objects)==0:
					print('no object')
				for object in objects:
					class_name = object.find('name').text.lower().strip()
					# we're only concerned with clases in our list
					if class_name in self._name_to_class:
						bbox = object.find('bndbox')
						# VID dataset format follows Matlab, in which indexes start from 0
						x1 = float(bbox.find('xmin').text) - 1
						y1 = float(bbox.find('ymin').text) - 1
						x2 = float(bbox.find('xmax').text) - 1
						y2 = float(bbox.find('ymax').text) - 1
						boxes.append([x1, y1, x2, y2])
						labels.append(self._class_to_ind[self._name_to_class[class_name]])
				boxes  = np.array(boxes, dtype=np.float32)
				labels = np.array(labels, dtype=np.int64)
				if self.transform:
					image, boxes, labels = self.transform(image, boxes, labels)
				if self.target_transform:
					boxes, labels = self.target_transform(boxes, labels)
				images.append(image)
				boxes_seq.append(boxes)
				labels_seq.append(labels)
			return images, boxes_seq, labels_seq
		
	def get_image(self, index):
		image_id = self.seq_list[index]
		images = self._read_image(image_id)
		if self.transform:
			for i in range(0,len(images)):
				images[i], _ = self.transform(images[i])
		return images

	def get_annotation(self, index):
		image_id = self.seq_list[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.seq_list)

	@staticmethod
	def _read_image_seq_ids(image_sets_file):
		seq_list = []
		with open(image_sets_file) as f:
			for line in f:
				seq_list.append(line.rstrip())
		return seq_list

	

#For only object classification and localization
class ImagenetDataset:

	def __init__(self, data, root, transform=None, target_transform=None, is_val=False, keep_difficult=False, label_file=None):
		"""Dataset for VID data.
		Args:
			data: the path of the ILSVRC2015 dataset, the directory contains the following sub-directories:
				Annotations, ImageSets, Data
			root: the path of root directory of cache
			transform : object of transform class
			target_transform : object of target_transform class 
		"""
		self.data = pathlib.Path(data)
		self.root = pathlib.Path(root)
		self.transform = transform
		self.target_transform = target_transform
		self.is_val = is_val
		if is_val:
			image_sets_file = "datasets/val_VID_list.txt"
		else:
			image_sets_file = "datasets/train_VID_list.txt"
		self.ids = ImagenetDataset._read_image_seq_ids(image_sets_file)
		
		logging.info("using default Imagenet VID classes.")
		self._classes_names = ['__background__',  # always index 0
					'airplane', 'antelope', 'bear', 'bicycle',
					'bird', 'bus', 'car', 'cattle',
					'dog', 'domestic_cat', 'elephant', 'fox',
					'giant_panda', 'hamster', 'horse', 'lion',
					'lizard', 'monkey', 'motorcycle', 'rabbit',
					'red_panda', 'sheep', 'snake', 'squirrel',
					'tiger', 'train', 'turtle', 'watercraft',
					'whale', 'zebra']
		self._classes_map = ['__background__',  # always index 0
						'n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049']

		self._name_to_class = {self._classes_map[i]: self._classes_names[i] for i in range(len(self._classes_names))}
		self._class_to_ind = {classes_name: i for i, classes_name in enumerate(self._classes_names)}
		self.db = self.gt_roidb() 

	def __getitem__(self, index):
		data = self.db[index]
		boxes = data['boxes']
		labels = data['labels']
		image = self._read_image(data['image'])
		if self.transform:
			image, boxes, labels = self.transform(image, boxes, labels)
		if self.target_transform:
			boxes, labels = self.target_transform(boxes, labels)
		return image, boxes, labels

	def gt_roidb(self):
		"""
		return ground truth image regions database
		:return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
		"""
		if self.is_val:
			cache_file = os.path.join(self.root, 'val_VID_gt_db.pkl')
		else:
			cache_file = os.path.join(self.root, 'train_VID_gt_db.pkl')
		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				roidb = pickle.load(fid)
			logging.info('gt roidb loaded from {}'.format(cache_file))
			return roidb

		gt_roidb = [self.load_vid_annotation(index) for index in range(0, len(self.ids))]
		with open(cache_file, 'wb') as fid:
			pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
		logging.info('wrote gt roidb to {}'.format(cache_file))

		return gt_roidb

	def load_vid_annotation(self, i):
		"""
		for a given index, load image and bounding boxes info from XML file
		:param index: index of a specific image
		:return: record['boxes', 'labels']
		"""
		index = self.ids[i]
		roi_rec = dict()
		roi_rec['image'] = self.image_path_from_index(index)
		if self.is_val:
			annotation_file = self.data / f"Annotations/VID/val/{index}.xml"
		else:
			annotation_file = self.data / f"Annotations/VID/train/{index}.xml"
		objects = ET.parse(annotation_file).findall("object")
		boxes = []
		labels = []
		for obj in objects:
			class_name = obj.find('name').text.lower().strip()
			# we're only concerned with clases in our list
			if class_name in self._name_to_class:
				bbox = obj.find('bndbox')

				# VID dataset format follows Matlab, in which indexes start from 0
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])
				labels.append(self._class_to_ind[self._name_to_class[class_name]])
		roi_rec['boxes'] = np.array(boxes, dtype=np.float32)
		roi_rec['labels'] = np.array(labels, dtype=np.int64)
		return roi_rec
	def image_path_from_index(self, image_id):
		"""
		given image index, find out full path
		:param index: index of a specific image
		:return: full path of this image
		"""
		if self.is_val:
			image_file = self.data / f"Data/VID/val/{image_id}.JPEG"
		else:
			image_file = self.data / f"Data/VID/train/{image_id}.JPEG"
		return image_file

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def _read_image_seq_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _read_image(self, image_file):
		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image
		


