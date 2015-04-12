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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str, help='output filename')

arguments = parser.parse_args()

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

valid_pixels_labels = []
valid_files = []
valid_files_count = 0
test_files_count = 0
superpixels = []
validationOriginalImage = []

for i in xrange(0,num_files):

    if file_labels[i] != TESTING_LABEL:

        fe = Feature()
        fe.loadImage(im_file_names[i])
        fe.loadSuperpixelImage(200, 10)
        #fe.loadSuperpixelFromFile(sp_file_names[i])
        fe.loadLabelImage(label_file_names[i])

        featureVectors = fe.getFeaturesVectors()
        labels = fe.getSuperPixelLabels()

        #Test purposes
        fe.getEdges()

        # store data
        if file_labels[i] == TRAINING_LABEL:
            train_labels = np.append(train_labels, labels, 0)
            if train_data==[]:
                train_data = featureVectors
            else:
                train_data = np.vstack((train_data,featureVectors))
        else:
            # get superpixel valid files
            superpixels.append(fe.getSuperpixelImage())
            validationOriginalImage.append(im_file_names[i])
            # these two lines need to be added into featureExtraction class
            valid_files = sp.getSuperValidFiles(fe.getSuperpixelImage(), valid_files_count, valid_files)
            valid_pixels_labels.append(sp.getPixelLabel(fe.getLabelImage()))
            valid_files_count += 1
            valid_labels = np.append(valid_labels, labels, 0)
            if valid_data==[]:
                valid_data = featureVectors
            else:
                valid_data = np.vstack((valid_data,featureVectors))

    else:
        test_files_count += 1
    sys.stdout.write('\r')
    sys.stdout.write('progress %2.2f%%' %(100.0*i/num_files))
    sys.stdout.flush()

print np.array(train_data).shape # show total number of superpixels


<<<<<<< HEAD
scipy.io.savemat(arguments.output_file, {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'im_file_names':im_file_names, 'sp_file_names':sp_file_names, 'label_file_names':label_file_names,'valid_pixels_labels':valid_pixels_labels,'valid_files':valid_files,'valid_files_count':valid_files_count,'superpixels':superpixels,'test_files_count':test_files_count}, oned_as='column')
=======
scipy.io.savemat('test_data.mat', {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'im_file_names':im_file_names, 'sp_file_names':sp_file_names, 'label_file_names':label_file_names,'valid_pixels_labels':valid_pixels_labels,'valid_files':valid_files,'valid_files_count':valid_files_count,'superpixels':superpixels,'test_files_count':test_files_count,'validationOriginalImage':validationOriginalImage}, oned_as='column')
>>>>>>> test
