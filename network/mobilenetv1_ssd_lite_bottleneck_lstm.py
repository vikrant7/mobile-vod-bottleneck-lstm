from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from .mobilenet import MobileNetV1
from torch.autograd import Variable
from .ssd import SSD
from .predictor import Predictor
from config import mobilenetv1_ssd_config as config

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
	"""
	return Sequential(
		Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
			   groups=in_channels, stride=stride, padding=padding),
		ReLU(),
		Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
	)

class ConvLSTMCell(nn.Module):
	def __init__(self, input_channels, hidden_channels):
		super(ConvLSTMCell, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		#self.kernel_size = kernel_size
		self.num_features = 4

		#self.padding = int((kernel_size - 1) / 2)
		self.Wy  = Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
		self.Wi  = Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, groups=self.hidden_channels, bias=False)  
		self.Wbi = Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbf = Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbc = Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbo = Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)

		self.Wci = None
		self.Wcf = None
		self.Wco = None

	def forward(self, x, h, c):
		y = torch.cat((x, h),1)
		i = self.Wy(y)
		b = self.Wi(i)
		ci = torch.sigmoid(self.Wbi(b) + c * self.Wci)
		cf = torch.sigmoid(self.Wbf(b) + c * self.Wcf)
		cc = cf * c + ci * torch.relu(self.Wbc(b))
		co = torch.sigmoid(self.Wbo(b) + cc * self.Wco)
		ch = co * torch.relu(cc)
		return ch, cc

	def init_hidden(self, batch_size, hidden, shape):
		if self.Wci is None:
			self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
			self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
			self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))#.cuda()
		else:
			assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
			assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
		return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),#.cuda(),
				Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))#.cuda()
				)

class ConvLSTM(nn.Module):
	def __init__(self, input_channels, hidden_channels, height, width, batch_size):#, step=1, effective_step=[1]):
		super(ConvLSTM, self).__init__()
		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		#self.kernel_size = kernel_size
		self.cell = ConvLSTMCell(self.input_channels, self.hidden_channels)
		(h, c) = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, shape=(height, width))
		self.hidden_state = h
		self.cell_state = c

	def forward(self, input):
		new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
		self.hidden_state = new_h
		self.cell_state = new_c
		return self.hidden_state



def create_mobilenetv1_ssd_bottleneck_lstm(num_classes, is_test=False):
	bottleneck = ConvLSTM(input_channels=1024, hidden_channels=256, height=10, width=10, batch_size=1)
	base_net = MobileNetV1(1001).model  # disable dropout layer

	source_layer_indexes = [
		12,
	]
	extras = ModuleList([
		Sequential(
			Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, groups=1024, stride=1, padding=1),
			bottleneck,
		),
		Sequential(	
			Conv2d(in_channels=256, out_channels=128, kernel_size=1),
			ReLU(inplace=True),
			SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
		),
		Sequential(
			Conv2d(in_channels=256, out_channels=64, kernel_size=1),
			ReLU(inplace=True),
			SeperableConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
		),
		Sequential(
			Conv2d(in_channels=128, out_channels=64, kernel_size=1),
			ReLU(inplace=True),
			SeperableConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
		),
		Sequential(
			Conv2d(in_channels=128, out_channels=32, kernel_size=1),
			ReLU(inplace=True),
			SeperableConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
		)
	])

	regression_headers = ModuleList([
		SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
		Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = ModuleList([
		SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
		Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
	])

	return SSD(num_classes, base_net, source_layer_indexes,
			   extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv1_ssd_bottleneck_lstm_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
	predictor = Predictor(net, config.image_size, config.image_mean,
						  config.image_std,
						  nms_method=nms_method,
						  iou_threshold=config.iou_threshold,
						  candidate_size=candidate_size,
						  sigma=sigma,
						  device=device)
	return predictor
