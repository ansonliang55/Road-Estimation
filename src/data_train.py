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
from sklearn import svm
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
scaler = StandardScaler()
#clf = SGDClassifier(loss="log", penalty="l2")
#clf = SGDRegressor(loss="squared_loss")
#clf = GaussianNB()
clf = svm.SVC(kernel='rbf', probability=True)
scaler.fit(train_data)
train_data = scaler.transform(train_data)

clf.fit(train_data, train_labels.ravel())

valid_data = scaler.transform(valid_data)
print clf.predict_proba(valid_data[0])
#wait = input("PRESS ENTER TO CONTINUE.")

count_correct=0
total_sample = len(valid_data)
for i in xrange(0,total_sample):
		if clf.predict(valid_data[i]) == valid_labels[i]:
				count_correct+=1

#print ('Validation Accuracy: %2.2f%%')%100.0*count_correct/total_sample


sp_file_names = data['sp_file_names'][0].strip()
im_file_names = data['im_file_names'][0].strip()

test_data, im_sp, image = fe.getFeaturesVectors(im_file_names, sp_file_names)

test_data = scaler.transform(test_data)
sp.showPrediction(clf, im_sp, test_data, image)



