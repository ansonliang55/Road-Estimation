import numpy as np
import cPickle
import scipy.io, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_db_path', help='Path to training database')
arguments = parser.parse_args()

data = scipy.io.loadmat(arguments.train_db_path)
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
train_edgesFeatures = data['train_edgesFeatures']
valid_edgesFeatures = data['valid_edgesFeatures']
valid_pixels_labels = data['valid_pixels_labels']
test_files_count = data['test_files_count']
validationOriginalImage = data['validationOriginalImage']


#print len(train_data)
#print len(valid_data)
#print len(train_labels)
#print len(valid_labels)
#print len(train_superpixels)
#print len(valid_superpixels)
#print len(train_edges)
#print type(train_edges)
#print len(valid_edges)
#print len(train_edgesFeatures)
#print len(valid_edgesFeatures)

trainSuperpixelNum = len(train_data)/len(train_edges)
train_X = []
for i in range (0, len(train_edges)):
    unary = []
    for j in range (0, trainSuperpixelNum):
        unary.append(train_data[i*trainSuperpixelNum + j])
    unary = np.array(unary)
    edges = []
    for j in range(0, trainSuperpixelNum):
        for k in range(j,trainSuperpixelNum):
        	if (train_edges[i])[j][k] == 1:
           		edges.append([j,k])
    edges = np.array(edges)
    edgesFeatures = np.array(train_edgesFeatures[i])
    train_X.append((unary,edges,edgesFeatures))

train_Y = []
for i in range (0, len(train_edges)):
    labels = np.zeros([1,trainSuperpixelNum],dtype = int)
    for j in range (0, trainSuperpixelNum):
        labels[0][j] = train_labels[i*trainSuperpixelNum + j]
    train_Y.append(labels[0])

data_train = {'X':train_X,'Y':train_Y}
cPickle.dump(data_train,open("data_train.pickle","wb"))

validSuperpixelNum = len(valid_data)/len(valid_edges)
valid_X = []
for i in range (0, len(valid_edges)):
    unary = []
    for j in range (0, validSuperpixelNum):
        unary.append(valid_data[i*validSuperpixelNum + j])
    unary = np.array(unary)
    edges = []
    for j in range(0, validSuperpixelNum):
        for k in range(j,validSuperpixelNum):
        	if (valid_edges[i])[j][k] == 1:
           		edges.append([j,k])
    edges = np.array(edges)
    edgesFeatures = np.array(valid_edgesFeatures[i])
    valid_X.append((unary,edges,edgesFeatures))
valid_Y = []
for i in range (0, len(valid_edges)):
    labels = np.zeros([1,validSuperpixelNum],dtype = int)
    for j in range (0, validSuperpixelNum):
        labels[0][j] = valid_labels[i*validSuperpixelNum + j]
    valid_Y.append(labels[0])
data_valid = {'X':valid_X,'Y':valid_Y}
cPickle.dump(data_valid,open("data_val_dict.pickle","wb"))
