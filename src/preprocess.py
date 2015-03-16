"""
Writen by: Anson Liang
"""

from scipy import misc
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import cv
import matplotlib.cm as cm



im = misc.imread('../data_road/training/image/um_000000.png',True)
target_im = misc.imread('../data_road/training/label/um_lane_000000.png', True)

mat = scipy.io.loadmat('../data_road/training/image/um_000000.mat')
label = mat['labels']

for i in xrange(0, len(label)):
		for j in xrange(0, len(label[0])):
				if label[i][j] %2 == 0:
						label[i][j] == 0
				else:
						label[i][j] == 200
overlay = im - (target_im-76.24500275)*8 - label*8
plt.imshow(overlay)
plt.show()

print mat['labels']