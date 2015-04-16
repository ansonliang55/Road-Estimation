"""
Written by: Anson Liang, Wenjie Zi
"""
from skimage import io, color
import superpixel as sp
import glob
import scipy.io, sys
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from featureExtract import Feature
#import maxflow
import argparse
import numpy as np
from pystruct.learners import NSlackSSVM
from pystruct import learners
from pystruct.models import GraphCRF
from pystruct.utils import SaveLogger
from time import time
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('train_db_path', help='Path to training database')
parser.add_argument('test_path', help='Path to test database')
arguments = parser.parse_args()
data = scipy.io.loadmat(arguments.train_db_path)
test = scipy.io.loadmat(arguments.test_path)
train_data = data['train_data']
valid_data = data['valid_data']
valid_edges = data['valid_edges']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
test_data = test['test_data']
test_labels = test['test_label']
test_edges = test['test_edges']
# Preprocessing normalize data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
#Preprocessing RandomizePCA
#pca = RandomizedPCA(n_components=15)
#pca.fit(train_data)

scaler.fit(valid_data)
valid_data = scaler.transform(valid_data)
scaler.fit(test_data)
test_data = scaler.transform(test_data)
#valid_data = pca.transform(valid_data)
clf = knn(n_neighbors=21, p=1)
clf = clf.fit(train_data,train_labels.ravel())
print clf.score(valid_data,valid_labels.ravel())
print clf.score(test_data,test_labels.ravel())
"""
for file_num in range(210,213):#test_files_count):
    # see test results
    sp_file_names = data['sp_file_names'][file_num].strip()
    im_file_names = data['im_file_names'][file_num].strip()

    # Extract features from image files
    fe = Feature()
    fe.loadImage(im_file_names)
    fe.loadSuperpixelImage()
    test_data = fe.getFeaturesVectors()
   # edges, feat = fe.getEdges()
    # Normalize data
    test_data = scaler.transform(test_data)
    #test_data = pca.transform(test_data)

    sp.showPrediction(file_num, clf, fe.getSuperpixelImage(), test_data, fe.getImage())
    """
valid_count = 0
#for i in range(0,len(valid_edges)):
#    print valid_edges[i][0].shape
x_valid = []
for i in range(0, valid_data.shape[0]):
    temp = np.zeros((1,2),dtype = int)
#    print clf.predict_proba(valid_data[i])[0]
    x_valid.append(clf.predict_proba(valid_data[i])[0])
unary_file = []
for i in range(0, len(valid_edges)):
    unary = []
    for j in range (0, valid_edges[i][0].shape[0]):
        unary.append(x_valid[valid_count + j])
    valid_count = valid_count + valid_edges[i][0].shape[0]
    unary = np.array(unary)
    unary_file.append(unary)
X_valid = []
for i in range(0,len(unary_file)):
    edges = []
    for j in range(0, (valid_edges[i][0]).shape[0]):
        for k in range(j,(valid_edges[i][0]).shape[1]):
           if  (valid_edges[i][0])[j][k] == 1:
               edges.append([j,k])
    edges = np.array(edges)
    X_valid.append((np.atleast_2d(unary_file[i]), np.array(edges, dtype=np.int)))
print len(X_valid)
print len(X_valid[0])
print type(X_valid[0][0])
print X_valid[0][0].shape
print type(X_valid[0][1])
print X_valid[0][1].shape
valid_Y = []
valid_count = 0
for i in range (0, len(valid_edges)):
    labels = np.zeros([1,valid_edges[i][0].shape[0]],dtype = int)
    for j in range (0, valid_edges[i][0].shape[0]):
        labels[0][j] = valid_labels[valid_count + j].astype(int)
    valid_count = valid_count + valid_edges[i][0].shape[0]
    valid_Y.append(labels[0])
x_test = []
test_count = 0
for i in range(0, test_data.shape[0]):
    x_test.append(clf.predict_proba(test_data[i])[0])
unary_file = []
for i in range(0, len(test_edges)):
    unary = []
    for j in range (0, test_edges[i][0].shape[0]):
        unary.append(x_test[(test_count + j)])
    test_count = test_count +test_edges[i][0].shape[0]
    unary = np.array(unary)
    unary_file.append(unary)
X_test = []
for i in range(0,len(unary_file)):
    edges = []
    for j in range(0, test_edges[i][0].shape[0]):
        for k in range (j,test_edges[i][0].shape[1]):
           if  (test_edges[i][0])[j][k] == 1:
               edges.append([j,k])
 #   edges = np.array(edges)
    X_test.append((np.atleast_2d(unary_file[i]), np.array(edges, dtype=np.int)))
test_count = 0
test_Y = []
for i in range (0, len(test_edges)):
    labels = np.zeros([1,test_edges[i][0].shape[0]],dtype = int)
    for j in range (0, test_edges[i][0].shape[0]):
        labels[0][j] = test_labels[test_count + j].astype(int)
    test_count = test_count + test_edges[i][0].shape[0]
    test_Y.append(labels[0])
"""    
x_test = []
for i in range(0, valid_data.shape[0]):
    temp = np.zeros((1,2),dtype = int)
    if clf.predict(valid_data[i]) == 0:
        temp[0][0] = 1
    else:
        temp[0][1] = 1
    x_test.append(temp[0])
X_test = [(x, np.empty((0, 2), dtype=np.int)) for x in x_test]
print len(x_test)
for i in range(len(test_labels)):
    test_labels = test_labels.astype(int)
"""
print len(test_labels)
pbl = GraphCRF(inference_method='ad3')
svm = NSlackSSVM(pbl, C=1,n_jobs = 1,verbose = 1)
start = time()
print len(X_valid)
print len(valid_Y)
svm.fit(X_valid, valid_Y)
print "fit finished"
time_svm = time() - start
print X_test[i][0].shape
print svm.score(X_valid,valid_Y)
print svm.score(X_test,test_Y)
y_pred = np.vstack(svm.predict(np.array(X_valid)))
print("Score with pystruct crf svm: %f (took %f seconds)"
      % (np.mean(y_pred == valid_Y), time_svm))
y_predt = np.vstack(svm.predict(np.array(X_test)))
print("Score with pystruct crf svm: %f (took %f seconds)"
      % (np.mean(y_predt == test_Y), time_svm))


#we throw away void superpixels and flatten everything
#y_pred, y_true = np.hstack(y_pred), np.hstack(valid_Y)
#y_pred = y_pred[y_true != 255]
#y_true = y_true[y_true != 255]

#print("Score on test set: %f" % np.mean(y_true == y_pred))