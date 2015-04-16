import numpy as np
import cPickle
import scipy.io, sys
import argparse
from sklearn.neighbors import KNeighborsClassifier as knn
parser = argparse.ArgumentParser()
parser.add_argument('train_db_path', help='Path to training database')
parser.add_argument('test_db_path', help='test to training database')
arguments = parser.parse_args()

data = scipy.io.loadmat(arguments.train_db_path)
test = scipy.io.loadmat(arguments.train_db_path)
train_data = data['train_data']
valid_data = data['valid_data']
train_labels = data['train_labels']
valid_labels = data['valid_labels']
valid_files = data['valid_files']
valid_files_count = data['valid_files_count']
train_superpixels = data['train_superpixels']
valid_superpixels = data['valid_superpixels']
train_edges = data['train_edges']
valid_edges = data['valid_edges']
train_edgesFeatures1 = data['train_edgesFeatures1']
valid_edgesFeatures1 = data['valid_edgesFeatures1']
train_edgesFeatures2 = data['train_edgesFeatures2']
valid_edgesFeatures2 = data['valid_edgesFeatures2']
valid_pixels_labels = data['valid_pixels_labels']
test_files_count = data['test_files_count']
validationOriginalImage = data['validationOriginalImage']
test_data = test['test_data']
test_labels = test['test_labels']
test_edges = data['valid_edges']
test_edgesFeatures1 = data['train_edgesFeatures1']
test_edgesFeatures1 = data['valid_edgesFeatures1']

start = time.clock()

# Preprocessing normalize data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)

# Preprocessing RandomizePCA
pca = RandomizedPCA(n_components=15)
pca.fit(train_data)
#train_data = pca.transform(train_data)
print train_data.shape

# set classifier and fit data
clf = chooseClassification('RF')
clf = clf.fit(train_data,train_labels.ravel())
#scores = cross_val_score(clf, train_data, train_label)
#scores.mean()

# benchmark using validation data
valid_data = scaler.transform(valid_data)
#valid_data = pca.transform(valid_data)
#print clf.predict_proba(valid_data[0])
#wait = input("PRESS ENTER TO CONTINUE.")
end = time.clock()
x_valid = []
for i in range(0, valid_data.shape[0]):
    temp = np.zeros((1,2),dtype = int)
    if clf.predict(valid_data[i]) == 0:
        temp[0][0] = 1
    else:
        temp[0][1] = 1
    x_valid.append(temp[0])
validSuperpixelNum = len(valid_data)/len(valid_edges)
valid_X = []
for i in range (0, len(valid_edges)):
    unary = []
    for j in range (0, validSuperpixelNum):
        unary.append(x_valid[i*validSuperpixelNum + j])
    unary = np.array(unary)
    edges = []
    for j in range(0, validSuperpixelNum):
        for k in range(j,validSuperpixelNum):
        	if (valid_edges[i][0])[j][k] == 1:
           		edges.append([j,k])
           		edgesFeatures.append([1.000,valid_edgesFeatures1[i][j][k],valid_edgesFeatures2[i][j][k]])
    edges = np.array(edges)
    valid_X.append((unary,edges,edgesFeatures))
valid_Y = []
for i in range (0, len(valid_edges)):
    labels = np.zeros([1,validSuperpixelNum],dtype = int)
    for j in range (0, validSuperpixelNum):
        labels[0][j] = valid_labels[i*validSuperpixelNum + j].astype(int)
    valid_Y.append(labels[0])
x_test = []
for i in range(0, test_data.shape[0]):
    temp = np.zeros((1,2),dtype = int)
    if clf.predict(test_data[i]) == 0:
        temp[0][0] = 1
    else:
        temp[0][1] = 1
    x_test.append(temp[0])
testSuperpixelNum = len(test_data)/len(test_edges)
test_X = []
for i in range (0, len(test_edges)):
    unary = []
    for j in range (0, testSuperpixelNum):
        unary.append(x_test[i*testSuperpixelNum + j])
    unary = np.array(unary)
    edges = []
    for j in range(0, testSuperpixelNum):
        for k in range(j,testSuperpixelNum):
            if (test_edges[i][0])[j][k] == 1:
                edges.append([j,k])
                edgesFeatures.append([1.000,test_edgesFeatures1[i][j][k],test_edgesFeatures2[i][j][k]])
    edges = np.array(edges)
    test_X.append((unary,edges,edgesFeatures))
test_Y = []
for i in range (0, len(valid_edges)):
    labels = np.zeros([1,validSuperpixelNum],dtype = int)
    for j in range (0, validSuperpixelNum):
        labels[0][j] = valid_labels[i*validSuperpixelNum + j].astype(int)
    valid_Y.append(labels[0])
scipy.io.savemat(arguments.output_file, {'x_valid':valid_X,'y_valid':valid_Y,'x_test':test_X,'y_test':test_Y}, oned_as='column')
