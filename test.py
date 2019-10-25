#!/usr/bin/python3
"""Script for testing various functions during implementation
"""
#import argparse
import os
#import logging
import sys
#import itertools

import torch
# from torch.utils.data import DataLoader, ConcatDataset
# from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# from utils.misc import str2bool, Timer, store_labels
from network.mvod_basenet import MobileVOD, SSD, MobileNetV1, MatchPrior
# from datasets.vid_dataset_new import ImagenetDataset
# from network.multibox_loss import MultiboxLoss
# from config import mobilenetv1_ssd_config
# from dataloaders.data_preprocessing import TrainAugmentation, TestTransform

if __name__ == '__main__':
	#logging.info("Loading weights from pretrained netwok")
	pretrained_net_dict = torch.load('models/mobilenetv1_new.pth',map_location=lambda storage, loc: storage)
	#new_keys = []
	for k,v in pretrained_net_dict.items():
		if k == 'pred_encoder.model.1.3.weight':
			print(k)
			print(v)

		
