"""
Writen by: Anson Liang
"""

from scipy import misc
import matplotlib.pyplot as plt


im = misc.imread('../data_road/training/image_2/um_000000.png')
plt.imshow(im)
plt.show()