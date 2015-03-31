"""
Written by: Anson Liang
"""

from skimage import io, color
from skimage.util import img_as_float
import superpixel as sp
import scipy.io, sys
import numpy as np
from sklearn.linear_model import SGDClassifier,SGDRegressor
import superpixel as sp
import slic as sl
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
#constant
TRAINING_LABEL=0
VALIDATION_LABEL=1
TESTING_LABEL=2

numSegments = 200
com_factor = 10

data = scipy.io.loadmat('data.mat')
train_data = data['train_data']
valid_data = data['valid_data']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
#scaler = StandardScaler()
#clf = SGDClassifier(loss="log", penalty="l2")
#clf = SGDRegressor(loss="squared_loss")
clf = GaussianNB()
#scaler.fit(train_data)
#train_data = scaler.transform(train_data)

clf.fit(train_data, train_labels.ravel())

#valid_data = scaler.transform(valid_data)
print clf.predict_proba(valid_data[0])
#wait = input("PRESS ENTER TO CONTINUE.")

count_correct=0
total_sample = len(valid_data)
for i in xrange(0,total_sample):
		if clf.predict(valid_data[i]) == valid_labels[i]:
				count_correct+=1

print 1.0*count_correct/total_sample
#print clf.coef_

sp_file_names = data['sp_file_names'][100].strip()
im_file_names = data['im_file_names'][100].strip()


# read input image
image = img_as_float(io.imread(im_file_names))

# get slic superpixel segmentation
im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
#im_sp = scipy.io.loadmat(sp_file_names)['labels']

# get superpixel centroid location
sp_location = sp.getSuperPixelLocations(im_sp)

# get superpixel mean color		
sp_color = sp.getSuperPixelMeanColor(im_sp, image)

# get superpixel size
sp_size = sp.getSuperPixelSize(im_sp)
		
test_data = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T
#test_data = scaler.transform(test_data)
sp.showPrediction(clf, im_sp, test_data, image)

#scipy.io.savemat('data.mat', {'train_data':train_data, 'valid_data':valid_data, 'train_labels':train_labels, 'valid_labels':valid_labels, 'file_labels':file_labels, 'im_file_names':im_file_names, 'sp_file_names':sp_file_names, 'label_file_names':label_file_names}, oned_as='column')

