import os
import numpy as np

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
		for image in image_list:
			file_write_obj.writelines(relative_path+'/'+image)
			file_write_obj.write('\n')

file_write_obj.close()
file_write_obj = open('val_VID_list.txt','w')
for dir in dirs_val:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		image_list = np.sort(os.listdir(seq_path))
		for image in image_list:
			file_write_obj.writelines(seq+'/'+image)
			file_write_obj.write('\n')

file_write_obj.close()
file_write_obj = open('test_VID_list.txt','w')
for dir in dirs_test:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		image_list = np.sort(os.listdir(seq_path))
		for image in image_list:
			file_write_obj.writelines(seq+image)
			file_write_obj.write('\n')

file_write_obj.close()
