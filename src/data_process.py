"""
Written by: Anson Liang
"""

from skimage import io, color
import slic as sl
from skimage.util import img_as_float
import superpixel as sp
import scipy.io
import numpy as np
import glob

numSegments = 200
com_factor = 10

im_file_names = glob.glob("../data_road/training/image/*.png")
mat_file_names = glob.glob("../data_road/training/image/*.mat")
label_file_names = glob.glob("../data_road/training/label/*road*.png")

sp_location_x = []
sp_location_y = []
sp_color_r = []
sp_color_g = []
sp_color_b = []
sp_label = []
sp_size = []

for i in xrange(0,len(im_file_names)):
		# read input image
		image = img_as_float(io.imread(im_file_names[i]))
		label_image = color.rgb2gray(io.imread(label_file_names[i]))#load in grayscale

		# get slic superpixel segmentation
		#im_sp = sl.getSlicSuperpixels(image, numSegments, com_factor)
		im_sp = scipy.io.loadmat(mat_file_names[i])['labels'] 

		# get superpixel centroid location
		sp_location = sp.getSuperPixelLocations(im_sp)
		sp_location_x = np.append(sp_location_x, sp_location.T[0], 0)
		sp_location_y = np.append(sp_location_y, sp_location.T[1], 0)

		# get superpixel mean color
		sp_color = sp.getSuperPixelMeanColor(im_sp, image)
		sp_color_r = np.append(sp_color_r, sp_color.T[0], 0)
		sp_color_g = np.append(sp_color_g, sp_color.T[1], 0)
		sp_color_b = np.append(sp_color_b, sp_color.T[2], 0)
		# get superpixel label
		sp_label = np.append(sp_label, sp.getSuperPixelLabel(im_sp, label_image, 0.5), 0)

		# get superpixel size
		sp_size = np.append(sp_size, sp.getSuperPixelSize(im_sp),0)

print np.array(sp_location_x).shape

data = (sp_location_x,sp_location_y,sp_color_r,sp_color_g,sp_color_b,sp_size)
data = np.array(data).T

scipy.io.savemat('data.mat', {'data':data, 'labels':sp_label}, oned_as='column')

