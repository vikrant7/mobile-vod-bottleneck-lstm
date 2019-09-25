import sys
import os
import argparse
import shutil
import h5py
import numpy as np
import pandas as pd
import scipy.misc as sp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob

args=None
'''
Base class for Parsing all the datasets
'''

class DatasetParser:
    def __init__(self, data_dir, _data_splits=[0.7, 0.1, 0.2]):

        self._classes = ['__background__',  # always index 0
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

        self._name_to_class = dict(zip(self._classes_map, self._classes))
        # Class name to index
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        # Structure to hold class statistics
        self._class_counter = dict()
        # Data location
        self.data_dir = data_dir
        # Initialize all the sub-directory structure following PASCAL VOC
        self.init_paths()
        # Data splits in the form of [train, val, test]
        self.data_splits = _data_splits
        assert sum(self.data_splits) == 1.0, "All the splits must sum to 1.0"

        # Rest of this data must be filled by class specific to each dataset
        self.dataset_name = None  # Prepend images files with this name
        self.im_list = None
        # To store information for all images, each image is a dictionary key
        # which stores a dictionary for each image containing class and
        # bounding box information
        self.im_data = dict()
        # Directory of image data
        self.im_dir = None

    def init_paths(self):
        '''
        Initialize all the paths under data_dir directory to replicate most of the Pascal VOC structure
        :return:
        '''
        # Assert that a valid data_dir is passed
        assert os.path.isdir(self.data_dir), "A valid directory required. No directory named {}".format(self.data_dir)
        return

class ImagenetVID(DatasetParser):
    def __init__(self, data_dir, dataset_path):
        # Calling the base class constructor first
        DatasetParser.__init__(self, data_dir)
        # Store the dataset path
        self.dataset_path = dataset_path
        self.unique_classes = []

        # Read the xml annotation files

        # Get all the images that are present in the dataset
        self.im_list = []
        self.img_to_annot_map = {}
        self.vid_list = {'train':{}, 'val':{}, 'test':{}}
        self.get_vid_list()

    def write_to_file(self):
        for segment in self.vid_list:
            fname = os.path.join(args.output_path, segment) + ".txt"
            if os.path.exists(fname):
                os.remove(fname)
            print("Writing to {}".format(fname))
            for video in self.vid_list[segment]:
                if len(self.vid_list[segment][video])==0: continue
                last_frame=int(self.vid_list[segment][video][-1].split('.')[0])
                for frame in self.vid_list[segment][video]:
                    frame_number = int(frame.split('.')[0])
                    with open(fname,"a+") as f:
                        f.write(os.path.join(segment,video,frame.split('.')[0]) + " 1" + " " + str(frame_number) + " " + str(last_frame) + "\n")
    def merge_train_val(self):
        raise NotImplementedError

    def get_vid_list(self):
        np.random.seed(1)
        # Iterate over train/val/test
        for segment in os.listdir(self.dataset_path):
            if segment not in self.vid_list: continue
            # Build list of video snippets for each segment
            seg_path = os.path.join(self.dataset_path, segment)
            n_frames = 0
            for i,vid in enumerate(os.walk(seg_path)):
                if i==0 or len(vid[2])==0:
                    print(vid[0])
                    continue
                frame_list = sorted(vid[2])
                #if os.path.basename(vid[0]) not in self.vid_list[segment]:
                #    self.vid_list[segment][os.path.basename(vid[0])]=[]
                #self.vid_list[segment][os.path.basename(vid[0])]=sorted(vid[2])
                if args.frames_per_video != -1:
                    frame_list = frame_list[0::int(np.ceil(len(frame_list) / float(args.frames_per_video)))]
                    # consecutive frame
                    # Sample starting frame
                    #if len(frame_list)>args.frame_per_video:
                    #    start_frame = np.random.choice(len(frame_list)-args.frames_per_video, size=1)[0]
                    #    frame_list = frame_list[start_frame:start_frame+int(args.frames_per_video)]
                    #else:
                    #    start_frame=0
                    #    frame_list = frame_list[start_frame:]

                n_frames += len(frame_list)
                if os.path.basename(vid[0]) not in self.vid_list[segment]:
                    self.vid_list[segment][os.path.basename(vid[0])]=[]
                self.vid_list[segment][os.path.basename(vid[0])] = frame_list


            print("Total frames in {}:{}".format(segment,n_frames))




def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Build ImageNet VID dataset.')
    parser.add_argument('--dataset', dest='dataset',
                        help='Name of the dataset',
                        default='ImagenetVID', choices=['ImagenetVID'],type=str)
    parser.add_argument('--input_path', dest='input_path',
                        help='Path to input video frames.',
                        default='./data/ILSVRC/Data/VID',
                        type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='Path to where the new data will be stored.',
                        default='./data/ILSVRC/ImageSets/VID',
                        type=str)
    parser.add_argument('--frames_per_video', dest='frames_per_video',
                        help='Number of frames to use per video. Default all.',
                        default=-1, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# To get the name of class from string
def str_to_classes(str):
    curr_class = None
    try:
        curr_class = getattr(sys.modules[__name__], str)
    except:
        print "Dataset class is not implemented"
    return curr_class

if __name__ == "__main__":

    args = parse_args()

    print('Called with args:')
    print(args)

    data_path = args.output_path
    datasets = [args.dataset]
    dataset_paths = [args.input_path]
    # Process all the datasets
    for dataset, dataset_path in zip(datasets, dataset_paths):
        curr_dataset = str_to_classes(dataset)(data_path, dataset_path)
        curr_dataset.write_to_file()