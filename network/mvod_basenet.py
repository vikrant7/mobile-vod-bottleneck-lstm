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


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
	"""
	return nn.Sequential(
		nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
			   groups=in_channels, stride=stride, padding=padding),
		nn.ReLU(),
		nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
	)

def conv_bn(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True)
			)
def conv_dw(inp, oup, stride):
			return nn.Sequential(
				nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
				nn.BatchNorm2d(inp),
				nn.ReLU(inplace=True),

				nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
				nn.ReLU(inplace=True),
			)
class MatchPrior(object):
	def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
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
		print("Initializing weights..")
		self._initialize_weights()
		#self.fc = nn.Linear(1024, num_classes)
	def _initialize_weights(self):
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
		x = self.model(x)
		return x


class SSD_FPN(nn.Module):
	def __init__(self,num_classes, alpha = 1, is_test=False, config = None):
		super(SSD_FPN, self).__init__()
		# Decoder
		self.is_test = is_test
		self.config = config
		self.num_classes = num_classes
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)
		self.conv13 = conv_dw(512*alpha, 1024*alpha, 2)
		#self.conv14 = conv_dw(1024*alpha,1024*alpha, 1)
		self.fmaps_1 = nn.Sequential(	
			nn.Conv2d(in_channels=1024*alpha, out_channels=256*alpha, kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=256*alpha, out_channels=512*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_2 = nn.Sequential(	
			nn.Conv2d(in_channels=512*alpha, out_channels=128*alpha, kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_3 = nn.Sequential(	
			nn.Conv2d(in_channels=256*alpha, out_channels=128*alpha, kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.fmaps_4 = nn.Sequential(	
			nn.Conv2d(in_channels=256*alpha, out_channels=128*alpha, kernel_size=1),
			nn.ReLU(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		self.regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=1),
		])

		self.classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=1),
		])

		print("Initializing weights..")
		self._initialize_weights()

	def _initialize_weights(self):
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
		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.view(confidence.size(0), -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.view(location.size(0), -1, 4)

		return confidence, location

	def forward(self, x):
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
	def __init__(self, pred_enc, pred_dec):
		super(MobileVOD, self).__init__()
		self.pred_encoder = pred_enc
		self.pred_decoder = pred_dec
		

	def forward(self, seq):
		x = self.pred_encoder(seq)
		confidences, locations = self.pred_decoder(x)
		return confidences , locations
		


