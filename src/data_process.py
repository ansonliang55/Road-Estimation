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

#constant
TRAINING_LABEL=0
VALIDATION_LABEL=1
TESTING_LABEL=2

numSegments = 200
com_factor = 10


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
				# read input image
				image = img_as_float(io.imread(im_file_names[i]))
				label_image = color.rgb2gray(io.imread(label_file_names[i]))#load in grayscale

				# get slic superpixel segmentation
				im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
				#im_sp = scipy.io.loadmat(sp_file_names[i])['labels'] 

				# get superpixel centroid location
				sp_location = sp.getSuperPixelLocations(im_sp)

				# get superpixel mean color		
				sp_color = sp.getSuperPixelMeanColor(im_sp, image)

				# get superpixel size
				sp_size = sp.getSuperPixelSize(im_sp)
		
				# store data
				if file_labels[i] == TRAINING_LABEL:
						# get superpixel label
						train_labels = np.append(train_labels, sp.getSuperPixelLabel(im_sp, label_image, 0.5), 0)
						#train_labels = np.append(train_labels, sp.getSuperPixelLabelPercent(im_sp, label_image), 0)
						if train_data==[]:
								train_data = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
						else:
								temp = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
								train_data = np.vstack((train_data,temp))
				else:
						valid_labels = np.append(valid_labels, sp.getSuperPixelLabel(im_sp, label_image, 0.5), 0)
						#valid_labels = np.append(valid_labels, sp.getSuperPixelLabelPercent(im_sp, label_image), 0)
						if valid_data==[]:
								valid_data = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
						else:
								temp = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
								valid_data = np.vstack((valid_data,temp))
		sys.stdout.write('\r')
		sys.stdout.write('progress %2.2f%%' %(100.0*i/num_files))
		sys.stdout.flush()

print np.array(train_data).shape # show total number of superpixels


scipy.io.savemat('data.mat', {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'im_file_names':im_file_names, 'sp_file_names':sp_file_names, 'label_file_names':label_file_names}, oned_as='column')

