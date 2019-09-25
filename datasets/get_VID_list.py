import os
import numpy as np

dirs = ['ILSVRC2015_VID_train_0000/',
		'ILSVRC2015_VID_train_0001/',
		'ILSVRC2015_VID_train_0002/',
		'ILSVRC2015_VID_train_0003/']
dirs_val = ['/media/sine/space/vikrant/ILSVRC2015/Data/VID/val/']
dirs_test = ['/media/sine/space/vikrant/ILSVRC2015/Data/VID/test/']
	



file_write_obj = open('train_VID_seqs_list.txt','w')
for dir in dirs:
	seqs = np.sort(os.listdir(os.path.join('/media/sine/space/vikrant/ILSVRC2015/Data/VID/train/'+dir)))
	for seq in seqs:
		seq_path = os.path.join('/media/sine/space/vikrant/ILSVRC2015/Data/VID/train/',dir,seq)
		relative_path = dir + seq
		image_list = np.sort(os.listdir(seq_path))
		for i in range(0,int(len(image_list)/10)):
			if (i+1)*10 <= len(image_list):
				line = relative_path +';'+image_list[10*i]+','+image_list[10*i+1]+','+image_list[10*i+2]+','+image_list[10*i+3]+','+image_list[10*i+4]+','+image_list[10*i+5]+','+image_list[10*i+6]+','+image_list[10*i+7]+','+image_list[10*i+8]+','+image_list[10*i+9]
				file_write_obj.writelines(line)
				file_write_obj.write('\n')

file_write_obj.close()
file_write_obj = open('val_VID_seqs_list.txt','w')
for dir in dirs_val:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		image_list = np.sort(os.listdir(seq_path))
		for i in range(0,int(len(image_list)/10)):
			if (i+1)*10 <= len(image_list):
				line = seq+';'+image_list[10*i]+','+image_list[10*i+1]+','+image_list[10*i+2]+','+image_list[10*i+3]+','+image_list[10*i+4]+','+image_list[10*i+5]+','+image_list[10*i+6]+','+image_list[10*i+7]+','+image_list[10*i+8]+','+image_list[10*i+9]
				file_write_obj.writelines(line)
				file_write_obj.write('\n')

file_write_obj.close()
file_write_obj = open('test_VID_seqs_list.txt','w')
for dir in dirs_test:
	seqs = np.sort(os.listdir(dir))
	for seq in seqs:
		seq_path = os.path.join(dir,seq)
		file_write_obj.writelines(seq_path)
		file_write_obj.write('\n')

file_write_obj.close()
