#!/usr/bin/python3
"""Script for training the MobileVOD with 2 LSTM layers. As in mobilenet, here we use depthwise seperable convolutions 
for reducing the computation without affecting accuracy much. Model is trained on Imagenet VID 2015 dataset.
Here we unroll LSTM for 10 steps and gives 10 consecutive frames of video as input.
Few global variables defined here are explained:
Global Variables
----------------
args : dict
	Has all the options for changing various variables of the model as well as hyper-parameters for training.
dataset : VIDDataset (torch.utils.data.Dataset, For more info see datasets/vid_dataset.py)
optimizer : optim.SGD
scheduler : CosineAnnealingLR, MultiStepLR (torch.optim.lr_scheduler)
config : mobilenetv1_ssd_config (See config/mobilenetv1_ssd_config.py for more info, where you can change input size and ssd priors)
loss : MultiboxLoss (See network/multibox_loss.py for more info)
"""
import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils.misc import str2bool, Timer, store_labels
from network.mvod_bottleneck_lstm2 import MobileVOD, SSD, MobileNetV1, MatchPrior
from datasets.vid_dataset import VIDDataset
from network.multibox_loss import MultiboxLoss
from config import mobilenetv1_ssd_config
from dataloaders.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(
	description='Mobile Video Object Detection (Bottleneck LSTM) Training With Pytorch')

parser.add_argument('--datasets', help='Dataset directory path')
parser.add_argument('--freeze_net', action='store_true',
					help="Freeze all the layers except the prediction head.")
parser.add_argument('--width_mult', default=1.0, type=float,
					help='Width Multiplifier')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
					help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
					help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
					help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
					help='initial learning rate for base net.')
parser.add_argument('--ssd_lr', default=None, type=float,
					help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--pretrained', help='Pre-trained model')
parser.add_argument('--resume', default=None, type=str,
					help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
					help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
					help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
					help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=1, type=int,
					help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
					help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
					help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
					help='Set the debug log output frequency.')
parser.add_argument('--sequence_length', default=10, type=int,
					help='sequence_length of video to unfold')
parser.add_argument('--use_cuda', default=True, type=str2bool,
					help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
					help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True
	logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1, sequence_length=10):
	net.train(True)
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	for i, data in enumerate(loader):
		images, boxes, labels = data
		for image, box, label in zip(images, boxes, labels):
			image = image.to(device)
			box = box.to(device)
			label = label.to(device)

			optimizer.zero_grad()
			confidence, locations = net(image)
			regression_loss, classification_loss = criterion(confidence, locations, label, box)  # TODO CHANGE BOXES
			loss = regression_loss + classification_loss
			loss.backward(retain_graph=True)
			optimizer.step()

			running_loss += loss.item()
			running_regression_loss += regression_loss.item()
			running_classification_loss += classification_loss.item()
		net.detach_hidden()
		if i and i % debug_steps == 0:
			avg_loss = running_loss / (debug_steps*sequence_length)
			avg_reg_loss = running_regression_loss / (debug_steps*sequence_length)
			avg_clf_loss = running_classification_loss / (debug_steps*sequence_length)
			logging.info(
				f"Epoch: {epoch}, Step: {i}, " +
				f"Average Loss: {avg_loss:.4f}, " +
				f"Average Regression Loss {avg_reg_loss:.4f}, " +
				f"Average Classification Loss: {avg_clf_loss:.4f}"
			)
			running_loss = 0.0
			running_regression_loss = 0.0
			running_classification_loss = 0.0
	net.detach_hidden()


def val(loader, net, criterion, device):
	net.eval()
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	num = 0
	for _, data in enumerate(loader):
		images, boxes, labels = data
		for image, box, label in zip (images, boxes, labels):
			image = image.to(device)
			box = box.to(device)
			label = label.to(device)
			num += 1

			with torch.no_grad():
				confidence, locations = net(image)
				regression_loss, classification_loss = criterion(confidence, locations, label, box)
				loss = regression_loss + classification_loss

			running_loss += loss.item()
			running_regression_loss += regression_loss.item()
			running_classification_loss += classification_loss.item()
		net.detach_hidden()
	return running_loss / num, running_regression_loss / num, running_classification_loss / num

def initialize_model(pred_enc, pred_dec):
	if args.pretrained:
		logging.info("Loading weights from pretrained netwok")
		pretrained_net_dict = torch.load(args.pretrained)

		model_dict = pred_enc.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_net_dict.items() if k in model_dict}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		pred_enc.load_state_dict(model_dict)

		model_dict = pred_dec.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_net_dict.items() if k in model_dict}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		pred_dec.load_state_dict(model_dict)



if __name__ == '__main__':
	timer = Timer()

	logging.info(args)
	config = mobilenetv1_ssd_config	#config file for priors etc.
	train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
	target_transform = MatchPrior(config.priors, config.center_variance,
								  config.size_variance, 0.5)

	test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

	logging.info("Prepare training datasets.")
	train_dataset = VIDDataset(args.datasets, transform=train_transform,
								 target_transform=target_transform)
	label_file = os.path.join("models/", "vid-model-labels.txt")
	store_labels(label_file, train_dataset._classes_names)
	num_classes = len(train_dataset._classes_names)
	logging.info(f"Stored labels into file {label_file}.")
	logging.info("Train dataset size: {}".format(len(train_dataset)))
	train_loader = DataLoader(train_dataset, args.batch_size,
							  num_workers=args.num_workers,
							  shuffle=True)
	logging.info("Prepare Validation datasets.")
	val_dataset = VIDDataset(args.datasets, transform=test_transform,
								 target_transform=target_transform, is_val=True)
	logging.info(val_dataset)
	logging.info("validation dataset size: {}".format(len(val_dataset)))

	val_loader = DataLoader(val_dataset, args.batch_size,
							num_workers=args.num_workers,
							shuffle=False)
	#num_classes = 30
	logging.info("Build network.")
	pred_enc = MobileNetV1(num_classes=num_classes, alpha = args.width_mult)
	pred_dec = SSD(num_classes=num_classes, batch_size=args.batch_size, alpha = args.width_mult, is_test=False)
	if args.resume is None:
		initialize_model(pred_enc, pred_dec)
		net = MobileVOD(pred_enc, pred_dec)
	else:
		net = MobileVOD(pred_enc, pred_dec)
		print("Updating weights from resume model")
		net.load_state_dict(
			torch.load(args.resume,
					   map_location=lambda storage, loc: storage))

	min_loss = -10000.0
	last_epoch = -1

	base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
	ssd_lr = args.ssd_lr if args.ssd_lr is not None else args.lr
	if args.freeze_net:
		logging.info("Freeze net.")
		for param in pred_enc.parameters():
			param.requires_grad = False
		#Freezing upto new lstm layer
		net.pred_decoder.conv13.requires_grad = False
		net.pred_decoder.bottleneck_lstm1.requires_grad = False
		net.pred_decoder.fmaps_1.requires_grad = False

	net.to(DEVICE)

	criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
							 center_variance=0.1, size_variance=0.2, device=DEVICE)
	optimizer = torch.optim.SGD([{'params': [param for name, param in net.pred_encoder.named_parameters()], 'lr': base_net_lr},
		{'params': [param for name, param in net.pred_decoder.named_parameters()], 'lr': ssd_lr},], lr=args.lr, momentum=args.momentum,
								weight_decay=args.weight_decay)
	logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
				 + f"Extra Layers learning rate: {ssd_lr}.")

	if args.scheduler == 'multi-step':
		logging.info("Uses MultiStepLR scheduler.")
		milestones = [int(v.strip()) for v in args.milestones.split(",")]
		scheduler = MultiStepLR(optimizer, milestones=milestones,
													 gamma=0.1, last_epoch=last_epoch)
	elif args.scheduler == 'cosine':
		logging.info("Uses CosineAnnealingLR scheduler.")
		scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
	else:
		logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
		parser.print_help(sys.stderr)
		sys.exit(1)

	logging.info(f"Start training from epoch {last_epoch + 1}.")
	for epoch in range(last_epoch + 1, args.num_epochs):
		scheduler.step()
		train(train_loader, net, criterion, optimizer,
			  device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, sequence_length=args.sequence_length)
		
		if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
			val_loss, val_regression_loss, val_classification_loss = val(val_loader, net, criterion, DEVICE)
			logging.info(
				f"Epoch: {epoch}, " +
				f"Validation Loss: {val_loss:.4f}, " +
				f"Validation Regression Loss {val_regression_loss:.4f}, " +
				f"Validation Classification Loss: {val_classification_loss:.4f}"
			)
			model_path = os.path.join(args.checkpoint_folder, f"lstm2-wm-{args.width_mult}/Epoch-{epoch}-Loss-{val_loss}.pth")
			torch.save(net.state_dict(), model_path)
			logging.info(f"Saved model {model_path}")
