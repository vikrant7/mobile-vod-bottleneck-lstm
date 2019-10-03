Mobile Video Object Detection
========================================

Code for the Paper

**[Mobile Video Object Detection with Temporally-Aware Feature Maps][1]**
Mason Liu, Menglong Zhu, CVPR 2018

<p align="center">
  <img src="lstm_ssd_intro.png" width=640 height=360>
</p>

\[[link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Mobile_Video_Object_CVPR_2018_paper.pdf)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:hq5rcMUUXysJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAXLdwXcU5g_wiMQ40EvbHQ9kTyvfUxffh&scisf=4&ct=citation&cd=-1&hl=en)\]

Introduction
------------

This paper introduces an online model for object detection in videos designed to run in real-time on low-powered mobile and embedded devices. Proposed approach combines fast single-image object detection with convolutional long short term memory (LSTM) layers to create an interweaved recurrent-convolutional architecture. 

Additionally, authors propose an efficient Bottleneck-LSTM layer that significantly reduces computational cost compared to regular LSTMs. This network achieves temporal awareness by using Bottleneck-LSTMs to refine and propagate feature maps across frames. 

This approach is substantially faster than existing detection methods in video, outperforming the fastest single-frame models in model size and computational cost while attaining accuracy comparable to much more expensive single-frame models on the Imagenet VID 2015 dataset. This model reaches a real-time inference speed of up to 15 FPS on a mobile CPU.

Dependencies
------------

1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+
4. torch-vision

Dataset 
-------

Download Imagenet VID 2015 dataset from \[[link](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php​)\]
To get list of training, validation and test dataset (make sure to change path of dataset in the scripts):
 * for basenet training run **`datasets/get_VID_list.py`** script.
 * for sequential training of LSTM layers run **`datasets/get_VID_seqs_list.py`** script.

**Note: Output of this scripts is already in the repo, so no need to run it again**
 

Two custom Pytorch Dataset classes are written in **`datasets/vid_dataset.py`** which ingests this dataset and
provides random batch / complete data while training and validation. One class is for basenet training while other class is for sequential training where unroll length of LSTM is 10 and 10 consecutive frames from video sequence are provided as input for the same. Here we are unrolling for 10 steps as mentioned in the paper. 


Training
--------

Make sure to be in python 3.6+ environment with all the dependencies installed.

As described in section 4.2 of the paper, model has two types of LSTM layers, one is Bottleneck LSTM layer which reduces the number of channels by 0.25 and the other is normal Conv LSTM which has same number of channels as output as that of input.

Training of multiple Conv LSTM layers is done in sequencial order i.e. fine tune and fix all the layers before the newly added LSTM layer.

Make sure to keep batch size same in lstm1, lstm2, lstm3, lstm4 and lstm5 training as the size of hidden and cell state of LSTM layers should be consistent while training. Also, make sure to keep width multiplier same.

By default, GPU is used for training. Here, freeze_net command line argument freezes the model as descriped in the paper. 

Before saving the checkpoint model, model gets validated on the validation set. All the checkpoint models are saved in **`models`** directory.

#### Basenet

Basenet is Mobilenet V1 with SSD. Train the basenet by executing following command: 

```sh
python train_mvod_basenet.py --datasets {path to ILSVRC2015 root dir} --batch_size 60 --num_epochs 30 --width_mult 1
```
If you want to train with any other width multiplier then change the command line argument width_mult accordingly.

For more help on command line args, execute the following command:

```sh
python train_mvod_basenet.py --help
```
#### Basenet with 1 Bottleneck LSTM 

As described in section 4.2 of the paper, first Bottleneck LSTM layer is placed after Conv13 layer and we freeze all the layers upto and including Conv13 layer.
To train model with one Bottleneck LSTM layer execute following command:

```sh
python train_mvod_lstm1.py --datasets {path to ILSVRC2015 root dir} --batch_size 10 --num_epochs 30 --pretrained {path to pretrained basenet model} --freeze_net True --width_mult 1 
```
Refer script docstring and inline comments in **`train_mvod_lstm1.py`** for understanding of execution.

#### Basenet with 2 Bottleneck LSTM

As described in section 4.2 of the paper, second Bottleneck LSTM layer is placed after Feature Map 1 layer and we freeze all the layers upto and including Feature Map 1 layer.
To train model with two LSTM layers execute following command:

```sh
python train_mvod_lstm2.py --datasets {path to ILSVRC2015 root dir} --batch_size 10 --num_epochs 30 --pretrained {path to pretrained basenet model} --freeze_net True --width_mult 1 
```
Refer script docstring and inline comments in **`train_mvod_lstm2.py`** for understanding of execution.


#### Basenet with 3 Bottleneck LSTM

As described in section 4.2 of the paper, third Bottleneck LSTM layer is placed after Feature Map 2 layer and we freeze all the layers upto and including Feature Map 2 layer.
To train model with three Bottleneck LSTM layers execute following command:

```sh
python train_mvod_lstm3.py --datasets {path to ILSVRC2015 root dir} --batch_size 10 --num_epochs 30 --pretrained {path to pretrained bottleneck lstm 2} --freeze_net True --width_mult 1 
```
Refer script docstring and inline comments in **`train_mvod_lstm3.py`** for understanding of execution.

#### Basenet with 3 Bottleneck LSTM and 1 LSTM

As described in section 4.2 of the paper, a LSTM layer is placed after Feature Map 3 layer and we freeze all the layers upto and including Feature Map 3 layer.
To train model with 3 Bottleneck LSTM layers and 1 LSTM layer execute following command:

```sh
python train_mvod_lstm4.py --datasets {path to ILSVRC2015 root dir} --batch_size 10 --num_epochs 30 --pretrained {path to pretrained bottleneck lstm 3} --freeze_net True --width_mult 1 
```
Refer script docstring and inline comments in **`train_mvod_lstm3.py`** for understanding of execution.

#### Basenet with 3 Bottleneck LSTM and 2 LSTM

As described in section 4.2 of the paper, second normal LSTM layer is placed after Feature Map 4 layer and we freeze all the layers upto and including Feature Map 4 layer.
To train model with 3 Bottleneck LSTM layers and 2 LSTM layer execute following command:

```sh
python train_mvod_lstm5.py --datasets {path to ILSVRC2015 root dir} --batch_size 10 --num_epochs 30 --pretrained {path to pretrained bottleneck lstm 4} --freeze_net True --width_mult 1 
```
Refer script docstring and inline comments in **`train_mvod_lstm3.py`** for understanding of execution.

Evaluation
----------


Evaluation script reports validation accuracy.


Results
-------
#### Main Results as in the paper:


|                                 | <sub>training data</sub>     | <sub>testing data</sub> | <sub>mAP@0.5</sub> |<sub>Params (M)</sub>|<sub>MAC (B)</sub>|
|---------------------------------|-------------------|--------------|---------|--------|--------|
| <sub>Bottleneck LSTM</br>(width_mult = 1)</sub>                    | <sub>ImageNet VID train</sub> | <sub>ImageNet VID validation</sub> | 54.4   | 3.24	| 1.13	|
| <sub>Bottleneck LSTM</br>(width_mult = 0.5)</sub>           | <sub>ImageNet VID train</sub> | <sub>ImageNet VID validation</sub> | 43.8    | 0.86	| 0.19	|

#### Reported metrics:

**TODO: Train model and report metric score** Due to limited GPU resource and the huge size of Imagenet VID 2015 dataset, training of the model is taking huge amount of time. I will report the metric score here once training is done. Update : I have trained Basenet and now training of lstm1  is going on.

References
----------

1. PyTorch Docs. [[http://pytorch.org/docs/master](http://pytorch.org/docs/master)]
2. PyTorch SSD [[https://github.com/qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)]


License
-------

BSD

[1]: https://arxiv.org/abs/1711.06368


