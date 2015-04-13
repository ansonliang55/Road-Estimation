import numpy as np
import cPickle

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger


data_train = cPickle.load(open("data_train.pickle"))
C = 0.01

x = data_train['X']
print type(x)
print len(x)
print type(x[0])
print len(x[0])
print type(x[0][0])
print x[0][0].shape
n_states = 2
print("number of samples: %s" % len(data_train['X']))
class_weights = 1. / np.bincount(np.hstack(data_train['Y']))
class_weights *= 2. / np.sum(class_weights)
print(class_weights)

model = crfs.EdgeFeatureGraphCRF(class_weight=class_weights,
                                 symmetric_edge_features=[0, 1],
                                 antisymmetric_edge_features=[2])

experiment_name = "edge_features_one_slack_trainval_%f" % C

ssvm = learners.NSlackSSVM(
    model, verbose=2, C=C, max_iter=1000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(data_train['X'], data_train['Y'])

data_val = cPickle.load(open("data_val_dict.pickle"))
y_pred = ssvm.predict(data_val['X'])

# we throw away void superpixels and flatten everything
y_pred, y_true = np.hstack(y_pred), np.hstack(data_val['Y'])
y_pred = y_pred[y_true != 255]
y_true = y_true[y_true != 255]

print("Score on validation set: %f" % np.mean(y_true == y_pred))