"""
Written by: Anson Liang, Wenjie Zi
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
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from sklearn.neighbors import KNeighborsClassifier as knn
import benchmark as bm
import time
from featureExtract import Feature
import networkx as nx
import matplotlib.pyplot as plt
#import maxflow


# function used for getting a classifiers
def chooseClassification(name):
    print "Choosen classfier:",name
    return {
        'NB': GaussianNB(),
        'KNN': knn(21),
        'ADA': adaBoost(n_estimators=100),
        'RF': rf(n_estimators = 30),
        'SVM': svm.SVC(kernel='rbf', probability=True),
        }.get(name, GaussianNB())    # default Gaussian Naive Bayes


#constant
TRAINING_LABEL=0
VALIDATION_LABEL=1
TESTING_LABEL=2

numSegments = 200
com_factor = 10

data = scipy.io.loadmat('test_data.mat')
train_data = data['train_data']
valid_data = data['valid_data']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
valid_files = data['valid_files']
valid_files_count = data['valid_files_count']
superpixels = data['superpixels']
valid_pixels_labels = data['valid_pixels_labels']
test_files_count = data['test_files_count']
#print valid_files

# record time used for training
start = time.clock()

# Preprocessing normalize data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)

# set classifier and fit data
clf = chooseClassification('RF')
clf = clf.fit(train_data,train_labels.ravel())
#scores = cross_val_score(clf, train_data, train_label)
#scores.mean()


# benchmark using validation data
valid_data = scaler.transform(valid_data)
#print clf.predict_proba(valid_data[0])
#wait = input("PRESS ENTER TO CONTINUE.")
end = time.clock()
print end
time = bm.countTime(start,end)
superpixelTotal = pixelTotal = superpixelCorrect = pixelCorrect = 0
for file_num in range(0, valid_files_count):
    temp1, temp2 = bm.accuracyOfSuperpixels(file_num,valid_files, valid_data, clf, valid_labels)
    temp3, temp4 = bm.accuracyOfPixels(file_num,valid_files, superpixels, valid_data, clf, valid_pixels_labels)
    superpixelCorrect = superpixelCorrect + temp1
    superpixelTotal = superpixelTotal + temp2
    pixelCorrect = pixelCorrect + temp3
    pixelTotal = pixelTotal +temp4
bm.overrallAverageResult(superpixelCorrect, superpixelTotal, pixelCorrect,pixelTotal)


for file_num in range(0,1):#test_files_count):
    # see test results
    sp_file_names = data['sp_file_names'][file_num].strip()
    im_file_names = data['im_file_names'][file_num].strip()

    # Extract features from image files
    fe = Feature()
    fe.loadImage(im_file_names)
    fe.loadSuperpixelImage(200, 10)
    test_data = fe.getFeaturesVectors()

    # Normalize data
    test_data = scaler.transform(test_data)

    sp.showPrediction(file_num, clf, fe.getSuperpixelImage(), test_data, fe.getImage())

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
#    G.add_edge(ind[i][0],ind[i][1], capacity=edgeValues[i])
pos = fe.getSuperpixelLocation()
nx.draw_networkx(G, pos=pos, with_labels=True)
plt.show()
"""





