"""
Written by: Anson Liang
"""

from skimage import io, color
import slic as sl
from skimage.util import img_as_float
import superpixel as sp
import scipy.io, sys
import numpy as np
import glob
import random
from featureExtract import Feature
#constant
TRAINING_LABEL=0
VALIDATION_LABEL=1
TESTING_LABEL=2


im_file_names = glob.glob("../data_road/training/image/*.png")
sp_file_names = glob.glob("../data_road/training/image/*.mat")
label_file_names = glob.glob("../data_road/training/label/*road*.png")

num_files = len(im_file_names)


# define data split
num_train = int(num_files * 0.6)
num_test = int(num_files * 0.3)
num_valid = num_files - num_train - num_test

# define data split label 0 - train, 1 - validation, 2 - test
file_labels = np.zeros(num_files)
for i in xrange(0,num_test):
		file_labels[i] = 2 
for i in xrange(num_test,(num_valid+num_test)):
		file_labels[i] = 1 

random.seed(42)
random.shuffle(file_labels)

train_labels = []
train_data = []
valid_labels = []
valid_data = []
for i in xrange(0,num_files):

		if file_labels[i] != TESTING_LABEL:

				fe = Feature()
				fe.loadImage(im_file_names[i])
				fe.loadSuperpixelImage(200, 10)
				#fe.loadSuperpixelFromFile(sp_file_names[i])
				fe.loadLabelImage(label_file_names[i])

				featureVectors= fe.getFeaturesVectors()
				labels = fe.getLabels()

				# store data
				if file_labels[i] == TRAINING_LABEL:
						train_labels = np.append(train_labels, labels, 0)
						if train_data==[]:
								train_data = featureVectors
						else:
								train_data = np.vstack((train_data,featureVectors))
				else:
						valid_labels = np.append(valid_labels, labels, 0)
						if valid_data==[]:
								valid_data = featureVectors
						else:
								valid_data = np.vstack((valid_data,featureVectors))
		sys.stdout.write('\r')
		sys.stdout.write('progress %2.2f%%' %(100.0*i/num_files))
		sys.stdout.flush()

print np.array(train_data).shape # show total number of superpixels


scipy.io.savemat('data.mat', {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'im_file_names':im_file_names, 'sp_file_names':sp_file_names, 'label_file_names':label_file_names}, oned_as='column')

