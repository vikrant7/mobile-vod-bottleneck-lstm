#!/usr/bin/python3
"""Script for creating basenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Tuple
from utils import box_utils
from collections import namedtuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
	Arguments:
		in_channels : number of channels of input
		out_channels : number of channels of output
		kernel_size : kernel size for depthwise convolution
		stride : stride for depthwise convolution
		padding : padding for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(in_channels), kernel_size=kernel_size,
			   groups=int(in_channels), stride=stride, padding=padding),
		nn.ReLU6(),
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=1),
	)

def conv_bn(inp, oup, stride):
	"""3x3 conv with batchnorm and relu
	Arguments:
		inp : number of channels of input
		oup : number of channels of output
		stride : stride for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
				nn.Conv2d(int(inp), int(oup), 3, stride, 1, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU6(inplace=True)
			)
def conv_dw(inp, oup, stride):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d having batchnorm and relu layers in between.
	Here kernel size is fixed at 3.
	Arguments:
		inp : number of channels of input
		oup : number of channels of output
		stride : stride for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
				nn.Conv2d(int(inp), int(inp), 3, stride, 1, groups=int(inp), bias=False),
				nn.BatchNorm2d(int(inp)),
				nn.ReLU6(inplace=True),

				nn.Conv2d(int(inp), int(oup), 1, 1, 0, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU6(inplace=True),
			)
class MatchPrior(object):
	"""Matches priors based on the SSD prior config
	Arguments:
		center_form_priors : priors generated based on specs and image size in config file
		center_variance : a float used to change the scale of center
		size_variance : a float used to change the scale of size
		iou_threshold : a float value of thresholf of IOU
	"""
	def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
		"""
		Arguments:
			gt_boxes : ground truth boxes
			gt_labels : ground truth labels
		Returns:
			locations of form (batch_size, num_priors, 4) and labels
		"""
		if type(gt_boxes) is np.ndarray:
			gt_boxes = torch.from_numpy(gt_boxes)
		if type(gt_labels) is np.ndarray:
			gt_labels = torch.from_numpy(gt_labels)
		boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
												self.corner_form_priors, self.iou_threshold)
		boxes = box_utils.corner_form_to_center_form(boxes)
		locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
		return locations, labels


def crop_like(x, target):
	"""
	Arguments:
		x : a tensor whose shape has to be cropped
		target : a tensor whose shape has to assert on x
	Returns:
		x having same shape as target
	"""
	if x.size()[2:] == target.size()[2:]:
		return x
	else:
		height = target.size()[2]
		width = target.size()[3]
		crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
		crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
	# fixed indexing for PyTorch 0.4
	return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


class MobileNetV1(nn.Module):
	def __init__(self, num_classes=1024, alpha=1):
		""" torch.nn.module for mobilenetv1 upto conv12
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
		"""
		super(MobileNetV1, self).__init__()
		# upto conv 12
		self.model = nn.Sequential(
			conv_bn(3, 32*alpha, 2),
			conv_dw(32*alpha, 64*alpha, 1),
			conv_dw(64*alpha, 128*alpha, 2),
			conv_dw(128*alpha, 128*alpha, 1),
			conv_dw(128*alpha, 256*alpha, 2),
			conv_dw(256*alpha, 256*alpha, 1),
			conv_dw(256*alpha, 512*alpha, 2),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			)
		logging.info("Initializing weights of base net")
		self._initialize_weights()
		#self.fc = nn.Linear(1024, num_classes)
	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x):
		"""
		Arguments:
			x : a tensor which is used as input for the model
		Returns:
			a tensor which is output of the model 
		"""
		x = self.model(x)
		return x


class SSD(nn.Module):
	def __init__(self,num_classes, alpha = 1, is_test=False, config = None):
		"""
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
			is_Test : a bool used to make model ready for testing
			config : a dict containing all the configuration parameters 
		"""
		super(SSD, self).__init__()
		# Decoder
		self.is_test = is_test
		self.config = config
		self.num_classes = num_classes
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)
		self.conv13 = conv_dw(512*alpha, 1024*alpha, 2)
		#self.conv14 = conv_dw(1024*alpha,1024*alpha, 1) #not using conv14 as mentioned in paper
		self.fmaps_1 = nn.Sequential(	
			nn.Conv2d(in_channels=int(1024*alpha), out_channels=int(256*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=256*alpha, out_channels=512*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_2 = nn.Sequential(	
			nn.Conv2d(in_channels=int(512*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_3 = nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_4 = nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=1),
		])

		self.classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=1),
		])

		logging.info("Initializing weights of SSD")
		self._initialize_weights()

	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def compute_header(self, i, x):
		"""
		Arguments:
			i : an int used to use particular classification and regression layer
			x : a tensor used as input to layers
		Returns:
			locations and confidences of the predictions
		"""
		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.view(confidence.size(0), -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.view(location.size(0), -1, 4)

		return confidence, location

	def forward(self, x):
		"""
		Arguments:
			x : a tensor which is used as input for the model
		Returns:
			confidences and locations of predictions made by model during training
			or
			confidences and boxes of predictions made by model during testing
		"""
		confidences = []
		locations = []
		header_index=0
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.conv13(x)
		#x = self.conv14(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_1(x)
		#x=self.bottleneck_lstm2(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_2(x)
		#x=self.bottleneck_lstm3(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_3(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.fmaps_4(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		confidences = torch.cat(confidences, 1)
		locations = torch.cat(locations, 1)
		
		if self.is_test:
			confidences = F.softmax(confidences, dim=2)
			boxes = box_utils.convert_locations_to_boxes(
				locations, self.priors, self.config.center_variance, self.config.size_variance
			)
			boxes = box_utils.center_form_to_corner_form(boxes)
			return confidences, boxes
		else:
			return confidences, locations

class MobileVOD(nn.Module):
	"""
		Module to join encoder and decoder of predictor model
	"""
	def __init__(self, pred_enc, pred_dec):
		"""
		Arguments:
			pred_enc : an object of MobilenetV1 class
			pred_dec : an object of SSD class
		"""
		super(MobileVOD, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		

	def forward(self, seq):
		"""
		Arguments:
			seq : a tensor used as input to the model  
		Returns:
			confidences and locations of predictions made by model
		"""
		x = self.pred_encoder(seq)
		confidences, locations = self.pred_decoder(x)
		return confidences , locations
		


