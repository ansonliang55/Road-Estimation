"""
Written by: Wenjie Zi

This is the handler of data preprocessing
"""

#@input: training data,
#@input: validation data
#@input: test data
#@output: the training data, validation data and test data after reducing dimensions
import numpy as np

def pca(n,test_data,train_data,valid_data):
    U, s, V = np.linalg.svd((train_data).T)
    Z = np.dot(U[:, :n], np.eye(n)*s[:n])
    Z = Z.T
    new_train = Z
    new_test = (np.dot(self.test_data.T, V[:n].T)).T
    new_valid = (np.dot(self.valid_data.T, V[:n].T)).T
    return new_train, new_valid, new_test