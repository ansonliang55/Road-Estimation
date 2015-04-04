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
import benchmark as bm
import slic as sl
from sklearn.preprocessing import StandardScaler
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifer as rf
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from sklearn.naive_bayes import GaussianNB
import time
#constant
start = time.clock()
TRAINING_LABEL=0
VALIDATION_LABEL=1
TESTING_LABEL=2

numSegments = 200
com_factor = 10
data = scipy.io.loadmat('test_data.mat')
train_data = data[  'train_data']
valid_data = data['valid_data']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
scaler = StandardScaler()
#Gaussian naive Bayes
#clf = SGDClassifier(loss="log", penalty="l2")
#clf = SGDRegressor(loss="squared_loss")
#clf = GaussianNB()

#clf = AdaBoostClassifier(n_estimators=100)
#scores = cross_val_score(clf, train_data, train_label)
# scores.mean()                             
#clf = rf(n_estimator = 30)
#clf = clf.fit(train_data,train_labels.revel())
#scores = cross_val_score(clf, train_data, train_labels)
#scores.mean()                             

clf = svm.SVC(kernel='rbf', probability=True)
scaler.fit(train_data)
train_data = scaler.transform(train_data)
clf.fit(train_data, train_labels.ravel())

valid_data = scaler.transform(valid_data)
#print clf.predict_proba(valid_data[0])
#wait = input("PRESS ENTER TO CONTINUE.")
end = time.clock()
time = bm.countTime(start,end)
superpixelAccu = accuracyOfSuperpixels(valid_data, clf, valid_labels)
pixelAccu = accuracyOfPixels(superpixels, valid_data, clf, valid_pixels_labels)

sp_file_names = data['sp_file_names'][55].strip()
im_file_names = data['im_file_names'][55].strip()
# read input image
image = img_as_float(io.imread(im_file_names))
# get slic superpixel segmentation
im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
#im_sp = scipy.io.loadmat(sp_file_names)['labels']
# get superpixel centroid location
sp_location = sp.getSuperPixelLocations(im_sp)
#get superpixel histogram of oriented gradient
sp_HOG = sp.getSuperPixelOrientedHistogram(im_sp, image)
# get superpixel mean color	
sp_color = sp.getSuperPixelMeanColor(im_sp, image)
# get superpixel color histogram	
sp_color = sp.getSuperPixelColorHistogram(im_sp, image)
# get superpixel size
sp_size = sp.getSuperPixelSize(im_sp)
test_data = np.vstack((sp_location.T,sp_color.T, sp_size.T)).T

test_data = scaler.transform(test_data)
sp.showPrediction(clf, im_sp, test_data, image)



