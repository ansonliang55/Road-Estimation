"""
Written by: Anson Liang
"""

from skimage import io, color
from skimage.util import img_as_float
import superpixel as sp
import scipy.io, sys
import numpy as np
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from featureExtract import Feature
import networkx as nx
import matplotlib.pyplot as plt
import maxflow
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
clf = GaussianNB()
#clf = svm.SVC(kernel='rbf', probability=True)
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

fe = Feature()
fe.loadImage(im_file_names)
fe.loadSuperpixelImage(200, 10)
test_data = fe.getFeaturesVectors()

test_data = scaler.transform(test_data)


"""
G=nx.Graph()
numSuperpixels = np.max(fe.getSuperpixelImage())+1
for i in xrange(0,numSuperpixels):
		G.add_node(i)#clf.predict_proba([test_data[i]])[0][1])

edges, edgeValues = fe.getEdges()

ind = np.where(edges != 0)

edgeValues = edgeValues[ind]
ind = zip(ind[0], ind[1])
print (ind[i][0],ind[i][1])
G.add_edges_from(ind, capacity=edgeValues)
#for i in xrange(0, len(ind)):
#		G.add_edge(ind[i][0],ind[i][1], capacity=edgeValues[i])
pos = fe.getSuperpixelLocation()
nx.draw_networkx(G, pos=pos, with_labels=True)
plt.show()
"""

sp.showPrediction(clf, fe.getSuperpixelImage(), test_data, fe.getImage())



